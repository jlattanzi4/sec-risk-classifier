import pytest

from pipeline.config import load_config


def test_reads_user_agent_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    (tmp_path / ".env").write_text("SEC_USER_AGENT=Jane Doe jane@example.com\n")
    cfg = load_config(root=tmp_path)
    assert cfg.sec_user_agent == "Jane Doe jane@example.com"
    assert cfg.raw_dir == tmp_path / "data/raw/edgar"
    assert cfg.interim_dir == tmp_path / "data/interim"


def test_missing_user_agent_names_the_fix(tmp_path, monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(SystemExit, match="SEC_USER_AGENT=Your Name you@example.com"):
        load_config(root=tmp_path)


def test_blank_user_agent_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "   ")
    with pytest.raises(SystemExit, match="SEC_USER_AGENT"):
        load_config(root=tmp_path)
