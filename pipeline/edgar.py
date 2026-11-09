"""Polite, cached downloads from SEC EDGAR.

SEC's fair-access policy asks for a User-Agent naming a contact and at most
10 requests a second. Every file is cached on disk, so a rerun makes no
requests for files it already has.
"""

import time
from pathlib import Path

import requests

RETRY_STATUSES = {429} | set(range(500, 600))
MAX_BACKOFF_SECONDS = 60
_last_request_at = 0.0


def _wait_for_turn(min_interval: float) -> None:
    """Sleep so successive requests start at least `min_interval` seconds apart."""
    global _last_request_at
    wait = _last_request_at + min_interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def fetch(
    url: str,
    dest: Path,
    user_agent: str,
    *,
    session: requests.Session | None = None,
    min_interval: float = 0.15,
    max_retries: int = 5,
) -> Path:
    """Download `url` to `dest` unless it's already there, and return `dest`.

    Retries 429 and 5xx responses with exponential backoff (1 s, doubling,
    capped at 60 s), up to `max_retries` attempts in all. The body is written
    to a `.part` file and renamed only when complete, so an interrupted
    download never leaves a truncated file in the cache.
    """
    if dest.exists():
        return dest
    session = session or requests.Session()
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        _wait_for_turn(min_interval)
        resp = session.get(url, headers={"User-Agent": user_agent}, stream=True, timeout=60)
        if resp.status_code in RETRY_STATUSES and attempt < max_retries - 1:
            resp.close()
            time.sleep(min(2**attempt, MAX_BACKOFF_SECONDS))
            continue
        resp.raise_for_status()
        _write_atomically(resp, dest)
        return dest
    raise AssertionError("unreachable: the last attempt returns or raises")


def _write_atomically(resp: requests.Response, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with open(part, "wb") as f:
            f.writelines(resp.iter_content(chunk_size=1 << 20))
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    finally:
        resp.close()
