import pytest

from articraft.agent._child_process import child_environment


@pytest.mark.parametrize(
    "name, value",
    [
        ("OPENAI_API_KEY", "sk-x"),
        ("MODAL_TOKEN_ID", "ak-x"),
        ("MODAL_TOKEN_SECRET", "as-x"),
        ("ROOMS_MODAL_TOKEN_SECRET", "as-x"),
        ("DATABASE_URL", "postgresql://user:pw@host:5432/db"),
        ("SOME_PASSWORD", "hunter2"),
        ("GCP_CREDENTIALS", "{}"),
    ],
)
def test_credentials_do_not_reach_model_authored_code(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    assert name not in child_environment()


@pytest.mark.parametrize(
    "name, value",
    [
        ("PATH", "/usr/bin"),
        ("HOME", "/root"),
        ("SPACEFORM_OUTPUT_DIR", "/tmp/out"),
        # A URL is not a secret unless it carries userinfo.
        ("SPACEFORM_API_URL", "https://api.example.com/v1"),
    ],
)
def test_ordinary_variables_are_kept(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    assert child_environment()[name] == value
