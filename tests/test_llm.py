import json

import pytest

from pipeline.llm import LLMError, parse_reply, run_batches

ITEMS = [{"id": f"s{i}", "text": f"sentence {i}"} for i in range(5)]


def good_reply(prompt: str) -> str:
    """Answer every item that appears in the prompt."""
    ids = [item["id"] for item in ITEMS if f'"{item["id"]}"' in prompt]
    return json.dumps([{"id": i, "label": "same"} for i in ids])


def runner(tmp_path, call, **kwargs):
    prompt = tmp_path / "prompt_v1.md"
    prompt.write_text("Label each item.")
    return run_batches(
        ITEMS, prompt_file=prompt, out_dir=tmp_path / "raw", model="haiku", batch_size=2,
        required=("label",), call=call, concurrency=1, **kwargs,
    )


def test_a_good_reply_is_saved_and_returned(tmp_path):
    calls = []

    def call(prompt, model):
        calls.append(model)
        return {"result": good_reply(prompt), "duration_ms": 12}

    out = runner(tmp_path, call)
    assert [o["id"] for o in out] == ["s0", "s1", "s2", "s3", "s4"]
    assert calls == ["haiku"] * 3                                   # 5 items in batches of 2
    request = json.loads((tmp_path / "raw/batch_000.request.json").read_text())
    assert request["ids"] == ["s0", "s1"] and request["model"] == "haiku"
    assert "Label each item." in request["prompt"] and "sentence 1" in request["prompt"]
    response = json.loads((tmp_path / "raw/batch_000.response.json").read_text())
    assert response["duration_ms"] == 12 and "s0" in response["result"]


@pytest.mark.parametrize("bad", [
    "I could not do this.",                                         # not JSON
    json.dumps([{"id": "s0", "label": "same"}]),                    # an id is missing
    json.dumps([{"id": "s0", "label": "same"}, {"id": "s1", "label": "same"},
                {"id": "s9", "label": "same"}]),                    # an id nobody asked for
    json.dumps([{"id": "s0", "label": "same"}, {"id": "s1"}]),      # a required key is missing
    json.dumps([{"id": "s0", "label": "same"}, {"id": "s0", "label": "same"}]),   # an id twice
    json.dumps({"id": "s0", "label": "same"}),                      # not an array
])
def test_a_bad_reply_is_retried_then_left_for_the_next_run(tmp_path, bad):
    attempts = []

    def call(prompt, model):
        first_batch = '"s0"' in prompt
        if first_batch:
            attempts.append(1)
        return {"result": bad if first_batch else good_reply(prompt)}

    out = runner(tmp_path, call, max_retries=2)
    assert len(attempts) == 3                                       # one try and two retries
    assert [o["id"] for o in out] == ["s2", "s3", "s4"]             # the other batches still count
    assert not (tmp_path / "raw/batch_000.response.json").exists()
    assert (tmp_path / "raw/batch_001.response.json").exists()


def test_a_second_run_makes_no_calls_for_batches_already_saved(tmp_path):
    runner(tmp_path, lambda prompt, model: {"result": good_reply(prompt)})

    def never(prompt, model):
        raise AssertionError("called again")

    assert len(runner(tmp_path, never)) == 5


def test_a_run_resumes_at_the_batch_that_failed(tmp_path):
    def flaky(prompt, model):
        return {"result": "no" if '"s2"' in prompt else good_reply(prompt)}

    assert len(runner(tmp_path, flaky, max_retries=0)) == 3
    asked = []

    def call(prompt, model):
        asked.append(prompt)
        return {"result": good_reply(prompt)}

    assert len(runner(tmp_path, call)) == 5
    assert len(asked) == 1 and '"s2"' in asked[0]


def test_saved_batches_for_other_items_are_refused(tmp_path):
    runner(tmp_path, lambda prompt, model: {"result": good_reply(prompt)})
    prompt = tmp_path / "prompt_v1.md"
    other = [{"id": f"x{i}", "text": "t"} for i in range(5)]
    with pytest.raises(LLMError, match="different items"):
        run_batches(other, prompt_file=prompt, out_dir=tmp_path / "raw", model="haiku", batch_size=2,
                    required=("label",), call=lambda p, m: {"result": "[]"}, concurrency=1)


def test_a_failed_call_stops_the_run(tmp_path):
    def limit(prompt, model):
        raise LLMError("usage limit reached")

    with pytest.raises(LLMError, match="usage limit"):
        runner(tmp_path, limit)
    assert not list((tmp_path / "raw").glob("*.response.json"))


def test_a_reply_in_a_code_fence_is_accepted():
    reply = 'Here you go:\n```json\n[{"id": "a", "label": "same"}]\n```'
    assert parse_reply(reply, ["a"], ("label",)) == [{"id": "a", "label": "same"}]


def test_replies_come_back_in_the_order_asked():
    reply = json.dumps([{"id": "b", "label": "x"}, {"id": "a", "label": "y"}])
    assert [o["id"] for o in parse_reply(reply, ["a", "b"], ("label",))] == ["a", "b"]


def test_an_empty_required_value_is_rejected():
    with pytest.raises(ValueError, match="label"):
        parse_reply(json.dumps([{"id": "a", "label": "  "}]), ["a"], ("label",))


def test_only_the_reply_and_how_it_was_made_are_kept():
    from pipeline.llm import public_record

    raw = {
        "result": "[]", "stop_reason": "end_turn", "duration_ms": 900, "is_error": False,
        "session_id": "abc", "uuid": "def", "credential_expires_at": "2026-12-01", "total_cost_usd": 0.01,
        "modelUsage": {"claude-haiku-5-5": {"inputTokens": 10, "outputTokens": 5, "costUSD": 0.01}},
    }
    assert public_record(raw) == {
        "result": "[]", "stop_reason": "end_turn", "duration_ms": 900,
        "models": {"claude-haiku-5-5": {"input_tokens": 10, "output_tokens": 5}},
    }
