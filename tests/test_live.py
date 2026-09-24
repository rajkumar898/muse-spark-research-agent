"""Real calls to the Meta Model API. Skipped unless RUN_LIVE_TESTS=1 (costs tokens)."""
import os

import pytest

import config
import tools
from meta_client import MetaClient

pytestmark = pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real API")


@pytest.fixture
def client():
    s = config.load_settings()
    if not s.api_key:
        pytest.skip("MODEL_API_KEY not set")
    return MetaClient(s.api_key, s.model, s.base_url, "low")


def test_live_text(client):
    result = client.respond([{"role": "user", "content": "Reply with the single word: ready"}])
    assert "ready" in result["text"].lower()


def test_live_tool_call(client):
    result = client.respond([{"role": "user", "content": "Use the calculator: percentage increase from 100 to 125."}],
                            tools=tools.tool_schemas(["calculate"]))
    assert result["tool_calls"] and result["tool_calls"][0]["name"] == "calculate"
