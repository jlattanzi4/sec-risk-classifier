import io

import pytest
import requests

from pipeline import edgar
from pipeline.edgar import fetch

URL = "https://www.sec.gov/Archives/edgar/full-index/2015/QTR1/form.idx"
UA = "Jane Doe jane@example.com"


class BrokenStream(io.RawIOBase):
    """A body that delivers its first chunk, then drops the connection."""

    def __init__(self, first: bytes):
        self.chunks = [first]

    def read(self, size=-1):
        if self.chunks:
            return self.chunks.pop()
        raise ConnectionResetError("connection dropped mid-download")


def response(status: int, body: bytes | io.RawIOBase = b"") -> requests.Response:
    r = requests.Response()
    r.status_code = status
    r.reason = {200: "OK", 429: "Too Many Requests", 503: "Service Unavailable"}[status]
    r.url = URL
    r.raw = body if isinstance(body, io.RawIOBase) else io.BytesIO(body)
    return r


class ScriptedSession:
    """Stands in for requests.Session: records each get and replays scripted responses."""

    def __init__(self, *responses: requests.Response):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, stream=False, timeout=None):
        self.calls.append({"url": url, "headers": headers or {}})
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(edgar.time, "sleep", lambda seconds: None)


def test_sends_user_agent_and_writes_file(tmp_path):
    dest = tmp_path / "2015" / "QTR1" / "form.idx"
    session = ScriptedSession(response(200, b"index body"))
    assert fetch(URL, dest, UA, session=session) == dest
    assert dest.read_bytes() == b"index body"
    assert session.calls == [{"url": URL, "headers": {"User-Agent": UA}}]


def test_cached_file_makes_no_request(tmp_path):
    dest = tmp_path / "form.idx"
    dest.write_bytes(b"cached")
    session = ScriptedSession()
    assert fetch(URL, dest, UA, session=session) == dest
    assert session.calls == []
    assert dest.read_bytes() == b"cached"


def test_retries_on_429_then_succeeds(tmp_path):
    dest = tmp_path / "form.idx"
    session = ScriptedSession(response(429), response(503), response(200, b"ok"))
    fetch(URL, dest, UA, session=session)
    assert len(session.calls) == 3
    assert dest.read_bytes() == b"ok"


def test_gives_up_after_max_retries(tmp_path):
    dest = tmp_path / "form.idx"
    session = ScriptedSession(*(response(503) for _ in range(5)))
    with pytest.raises(requests.HTTPError):
        fetch(URL, dest, UA, session=session, max_retries=5)
    assert len(session.calls) == 5
    assert not dest.exists()


def test_failed_write_leaves_no_partial_file(tmp_path):
    dest = tmp_path / "form.idx"
    session = ScriptedSession(response(200, BrokenStream(b"first half")))
    with pytest.raises(ConnectionResetError):
        fetch(URL, dest, UA, session=session)
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_successive_requests_are_spaced_by_min_interval(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(edgar.time, "sleep", sleeps.append)
    monkeypatch.setattr(edgar.time, "monotonic", lambda: 1000.0)  # a frozen clock
    monkeypatch.setattr(edgar, "_last_request_at", 0.0)  # no request made yet
    session = ScriptedSession(response(200, b"a"), response(200, b"b"))
    fetch(URL, tmp_path / "a.idx", UA, session=session, min_interval=0.15)
    fetch(URL, tmp_path / "b.idx", UA, session=session, min_interval=0.15)
    assert sleeps == [pytest.approx(0.15)]
