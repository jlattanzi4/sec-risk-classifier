"""Paths and settings for the pipeline, with the SEC contact read from .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[1]
MISSING_USER_AGENT = (
    "SEC requires a contact in every request's User-Agent. "
    "Add this line to .env: SEC_USER_AGENT=Your Name you@example.com"
)


@dataclass(frozen=True)
class Config:
    root: Path
    raw_dir: Path
    interim_dir: Path
    sec_user_agent: str


def load_config(env_file: Path | None = None, root: Path | None = None) -> Config:
    """Build the config, stopping before any network call if the SEC contact is missing.

    A SEC_USER_AGENT already set in the environment wins over the one in .env.
    """
    root = root or REPO_ROOT
    env_file = env_file or root / ".env"
    if "SEC_USER_AGENT" in os.environ:
        user_agent = os.environ["SEC_USER_AGENT"]
    else:
        user_agent = dotenv_values(env_file).get("SEC_USER_AGENT")
    if not user_agent or not user_agent.strip():
        raise SystemExit(MISSING_USER_AGENT)
    return Config(
        root=root,
        raw_dir=root / "data/raw/edgar",
        interim_dir=root / "data/interim",
        sec_user_agent=user_agent.strip(),
    )
