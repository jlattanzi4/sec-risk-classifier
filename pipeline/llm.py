"""Batched, resumable LLM calls through Claude Code's headless mode.

The project uses no API credits: every model call is `claude -p` on the
owner's Claude plan. Each batch's full prompt and the CLI's raw reply are
saved, so the labels can be audited, and reused without calling a model again.
A run that is interrupted (a usage limit, a closed laptop) picks up at the
first batch with no saved reply.
"""

import json
import re
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from pathlib import Path

SYSTEM = (
    "You are a careful annotator of sentences from corporate annual reports. "
    "Follow the instructions in the message exactly and reply with JSON only."
)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    """The model could not be called, or saved batches don't belong to this request."""


def public_record(reply: dict) -> dict:
    """What is worth keeping from the CLI's reply: the text, and how it was made.

    The CLI also reports session ids, account details and costs, which don't
    belong in a public repository.
    """
    models = {
        name: {"input_tokens": use.get("inputTokens"), "output_tokens": use.get("outputTokens")}
        for name, use in reply.get("modelUsage", {}).items()
    }
    kept = {key: reply[key] for key in ("result", "stop_reason", "duration_ms") if key in reply}
    return {**kept, "models": models}


def claude_cli(prompt: str, model: str, timeout: int = 900) -> dict:
    """One headless call. Returns the reply (`result`) with the model and token counts.

    Runs from an empty directory with tools, MCP servers, settings and slash
    commands off, so nothing from this machine's setup reaches the model.
    """
    command = [
        "claude", "-p", "--model", model, "--output-format", "json", "--system-prompt", SYSTEM,
        "--tools", "", "--strict-mcp-config", "--setting-sources", "",
        "--disable-slash-commands", "--no-session-persistence",
    ]  # fmt: skip
    with tempfile.TemporaryDirectory() as empty:
        proc = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=empty,
            timeout=timeout,
            check=False,   # a failure is reported below, with what the CLI printed
        )
    if proc.returncode != 0:
        raise LLMError(f"claude exited {proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}")
    reply = json.loads(proc.stdout)
    if reply.get("is_error"):
        raise LLMError(str(reply.get("result"))[:500])
    return public_record(reply)


def parse_reply(text: str, ids: list[str], required: tuple[str, ...]) -> list[dict]:
    """The reply as a list of objects, one per requested id, in the order asked.

    Raises ValueError unless it is a JSON array with exactly the requested ids,
    each once, each with a non-empty value for every required key.
    """
    fenced = _FENCE.search(text)
    body = fenced.group(1) if fenced else text
    start, end = body.find("["), body.rfind("]")
    if start < 0 or end < start:
        raise ValueError("no JSON array in the reply")
    rows = json.loads(body[start : end + 1])
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("the reply is not an array of objects")
    got = [str(r.get("id")) for r in rows]
    if sorted(got) != sorted(ids):
        raise ValueError(f"asked for {len(ids)} ids, got {len(got)} that don't match them")
    for row in rows:
        for key in required:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"item {row.get('id')} has no {key}")
    by_id = {str(r["id"]): r for r in rows}
    return [by_id[i] for i in ids]


def run_batches(
    items: list[dict],
    *,
    prompt_file: Path,
    out_dir: Path,
    model: str,
    batch_size: int,
    required: tuple[str, ...],
    call: Callable[[str, str], dict] = claude_cli,
    max_retries: int = 2,
    concurrency: int = 4,
) -> list[dict]:
    """Send `items` to the model in batches and return the parsed replies.

    Batch n's prompt is the text of `prompt_file` followed by the batch as
    JSON; it is saved as `batch_NNN.request.json`, and the CLI's reply as
    `batch_NNN.response.json` once it parses. A batch with a saved reply is
    never sent again. A reply that doesn't parse is retried `max_retries`
    times, then the batch is left without a reply for the next run, and the
    other batches still count. A call that fails outright stops the run.
    """
    instructions = prompt_file.read_text().strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = [list(b) for b in batched(items, batch_size)]
    results: dict[int, list[dict]] = {}
    pending = []
    for n, batch in enumerate(batches):
        ids = [str(item["id"]) for item in batch]
        request = out_dir / f"batch_{n:03d}.request.json"
        response = out_dir / f"batch_{n:03d}.response.json"
        if request.exists() and json.loads(request.read_text())["ids"] != ids:
            raise LLMError(
                f"{request} was made for different items; move {out_dir} away to start over"
            )
        prompt = f"{instructions}\n\n{json.dumps(batch, ensure_ascii=False, indent=1)}"
        if response.exists():
            results[n] = parse_reply(json.loads(response.read_text())["result"], ids, required)
            continue
        request.write_text(
            json.dumps(
                {"model": model, "prompt_file": prompt_file.name, "ids": ids, "prompt": prompt},
                ensure_ascii=False,
                indent=1,
            )
        )
        pending.append((n, ids, prompt, response))

    def send(job: tuple[int, list[str], str, Path]) -> tuple[int, list[dict] | None]:
        n, ids, prompt, response = job
        for _ in range(1 + max_retries):
            reply = call(prompt, model)
            try:
                parsed = parse_reply(reply["result"], ids, required)
            except (ValueError, json.JSONDecodeError):
                continue
            response.write_text(json.dumps(reply, ensure_ascii=False, indent=1))
            return n, parsed
        return n, None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for n, parsed in pool.map(send, pending):
            if parsed is not None:
                results[n] = parsed
    return [row for n in sorted(results) for row in results[n]]
