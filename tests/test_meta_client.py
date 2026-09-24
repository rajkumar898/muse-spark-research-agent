"""meta_client tests. The live client talks to httpx.MockTransport, never the real API."""
import json

import httpx
import pytest

import meta_client
from meta_client import MetaAPIError, MetaClient, MockMetaClient, ResilientClient, parse_response

KEY = "test-secret-key-123"

RESPONSE_WITH_TOOL = {
    "id": "resp_1", "object": "response", "model": "muse-spark-1.3",
    "output": [
        {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAA..."},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "calculate",
         "arguments": "{\"expression\": \"1+1\"}", "status": "completed"},
    ],
    "usage": {"input_tokens": 50, "output_tokens": 30, "output_tokens_details": {"reasoning_tokens": 20}},
}

RESPONSE_WITH_TEXT = {
    "id": "resp_2", "object": "response", "model": "muse-spark-1.3",
    "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [
        {"type": "output_text", "text": "Answer [web].", "annotations": [
            {"type": "url_citation", "url": "https://a.org", "title": "A", "start_index": 0, "end_index": 6},
            {"type": "url_citation", "url": "https://a.org", "title": "A", "start_index": 0, "end_index": 6},
        ]}]}],
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


def test_parse_tool_call_response():
    r = parse_response(RESPONSE_WITH_TOOL)
    assert r["text"] == ""
    assert r["tool_calls"] == [{"call_id": "call_1", "name": "calculate", "arguments": "{\"expression\": \"1+1\"}"}]
    assert r["usage"] == {"input_tokens": 50, "output_tokens": 30, "reasoning_tokens": 20}
    assert [i["type"] for i in r["output_items"]] == ["reasoning", "function_call"]   # kept for replay


def test_parse_text_and_citations():
    r = parse_response(RESPONSE_WITH_TEXT)
    assert r["text"] == "Answer [web]." and r["tool_calls"] == []
    assert r["citations"] == [{"title": "A", "url": "https://a.org"}]              # de-duplicated


@pytest.mark.parametrize("bad", [None, {}, {"output": "nope"},
                                 {"output": [{"type": "function_call", "name": "x"}]}])
def test_unexpected_shape(bad):
    with pytest.raises(MetaAPIError) as e:
        parse_response(bad)
    assert e.value.kind == "bad_shape"


def test_mock_uses_same_shape_as_real():
    r = MockMetaClient().respond([{"role": "user", "content": "Calculate the percentage increase from 100 to 125."}],
                                 tools=[{"type": "function", "name": "calculate"}])
    assert set(r) == {"text", "tool_calls", "citations", "usage", "output_items"}
    assert r["tool_calls"][0]["name"] == "calculate"
    assert json.loads(r["tool_calls"][0]["arguments"]) == {"expression": "pct_change(100, 125)"}


# ---------- live client against a fake HTTP server ----------

def make_client(handler, sleeps=None):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return MetaClient(KEY, "muse-spark-1.3", "https://api.meta.ai/v1", "low", http_client=http,
                      sleep=(sleeps.append if sleeps is not None else lambda s: None))


def error(status, message, type_="invalid_request_error", code=None, headers=None):
    return httpx.Response(status, headers=headers,
                          json={"error": {"message": message, "type": type_, "param": None, "code": code}})


def test_request_format_matches_meta_docs():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RESPONSE_WITH_TEXT)

    client = make_client(handler)
    tools = [{"type": "function", "name": "calculate", "description": "d", "parameters": {"type": "object"}}]
    result = client.respond([{"role": "user", "content": "hi"}], tools=tools, use_web_search=True, instructions="be brief")
    assert seen["url"] == "https://api.meta.ai/v1/responses"
    assert seen["auth"] == f"Bearer {KEY}"
    body = seen["body"]
    assert body["model"] == "muse-spark-1.3"
    assert body["reasoning"] == {"effort": "low"}
    assert body["instructions"] == "be brief"
    assert body["tools"] == tools + [{"type": "web_search"}]
    assert body["include"] == ["reasoning.encrypted_content"] and body["store"] is False
    assert result["text"] == "Answer [web]."


@pytest.mark.parametrize("response, kind, words", [
    (error(401, "Unauthorized", "authentication_error", "invalid_api_key"), "auth", "authentication failed"),
    (error(404, "model not found", code="model_not_found"), "model_not_found", "MUSE_MODEL"),
    (error(403, "This service is not available in your country"), "region", "region"),
    (error(403, "Forbidden"), "access", "no access"),
    (error(402, "no billing", "billing_error", "billing_not_configured"), "billing", "billing"),
    (error(400, "bad tool_choice"), "bad_request", "bad tool_choice"),
])
def test_errors_become_friendly_messages(response, kind, words):
    client = make_client(lambda request: response)
    with pytest.raises(MetaAPIError) as e:
        client.respond([{"role": "user", "content": "hi"}])
    assert e.value.kind == kind and words in str(e.value)
    assert KEY not in str(e.value)


def test_429_is_retried_with_backoff_then_succeeds():
    responses = iter([error(429, "slow down", "rate_limit_error", "rate_limit_exceeded"),
                      error(429, "slow down", "rate_limit_error", "rate_limit_exceeded", headers={"Retry-After": "7"}),
                      httpx.Response(200, json=RESPONSE_WITH_TEXT)])
    sleeps = []
    result = make_client(lambda r: next(responses), sleeps).respond([{"role": "user", "content": "hi"}])
    assert result["text"] == "Answer [web]."
    assert len(sleeps) == 2 and 1 <= sleeps[0] < 2 and sleeps[1] == 7


def test_429_gives_up_after_max_retries():
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        return error(429, "slow down", "rate_limit_error")

    with pytest.raises(MetaAPIError) as e:
        make_client(handler, sleeps).respond([{"role": "user", "content": "hi"}])
    assert e.value.kind == "rate_limit"
    assert len(calls) == meta_client.MAX_RETRIES + 1 and len(sleeps) == meta_client.MAX_RETRIES


def test_timeout_is_retried_then_reported():
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(MetaAPIError) as e:
        make_client(handler).respond([{"role": "user", "content": "hi"}])
    assert e.value.kind == "timeout"


def test_401_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return error(401, "Unauthorized", "authentication_error")

    with pytest.raises(MetaAPIError):
        make_client(handler).respond([{"role": "user", "content": "hi"}])
    assert len(calls) == 1


def test_missing_key():
    with pytest.raises(MetaAPIError) as e:
        MetaClient("", "muse-spark-1.3", "https://api.meta.ai/v1")
    assert "MODEL_API_KEY" in str(e.value)


def test_resilient_client_falls_back_to_mock_on_auth_error():
    live = make_client(lambda r: error(401, "Unauthorized", "authentication_error"))
    client = ResilientClient(live=live)
    result = client.respond([{"role": "user", "content": "hi"}])
    assert client.mode == "mock" and "authentication failed" in client.fallback_error
    assert "Mock Muse Spark" in result["text"]


def test_resilient_client_does_not_hide_other_errors():
    live = make_client(lambda r: error(400, "bad request"))
    with pytest.raises(MetaAPIError):
        ResilientClient(live=live).respond([{"role": "user", "content": "hi"}])


def test_make_client_modes():
    from config import Settings
    assert meta_client.make_client(Settings(api_key="")).mode == "mock"
    assert meta_client.make_client(Settings(api_key="k", force_mock=True)).mode == "mock"
    assert meta_client.make_client(Settings(api_key="k")).mode == "live"
    assert "k-secret" not in repr(Settings(api_key="k-secret"))


# ---------- mock routing: what the scripted stand-in "understands" ----------

AGENT_TOOLS = [{"type": "function", "name": n} for n in ("search_papers", "read_pdf", "calculate", "generate_report")]


def first_decision(question):
    return MockMetaClient().respond([{"role": "user", "content": question}], tools=AGENT_TOOLS)


@pytest.mark.parametrize("question, expression", [
    ("Calculate the percentage increase from 100 to 125.", "pct_change(100, 125)"),
    ("Divide 100 by 4", "100 / 4"),
    ("what is 100 divided by 4?", "100 / 4"),
    ("What is 15% of 80?", "15 / 100 * 80"),
    ("multiply 6 by 7", "6 * 7"),
    ("Calculate 144 / 12", "144 / 12"),
    ("what is (3 + 4) * 2?", "(3 + 4) * 2"),
    ("2^10", "2**10"),
])
def test_mock_understands_maths(question, expression):
    call = first_decision(question)["tool_calls"][0]
    assert call["name"] == "calculate" and json.loads(call["arguments"]) == {"expression": expression}


def test_mock_searches_only_for_research_requests():
    assert first_decision("Find papers about battery degradation")["tool_calls"][0]["name"] == "search_papers"
    assert first_decision("Review of studies from 2020-2024 on solar")["tool_calls"][0]["name"] == "search_papers"


@pytest.mark.parametrize("question", ["can you do division task?", "write an essay about dogs", "hello"])
def test_mock_explains_itself_instead_of_searching(question):
    r = first_decision(question)
    assert r["tool_calls"] == [] and "Divide 100 by 4" in r["text"]
