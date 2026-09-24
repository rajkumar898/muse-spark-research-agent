"""Thin wrapper around the Meta Model API (Muse Spark), plus a scripted MockMetaClient.

The Meta Model API is OpenAI-SDK compatible, so we use the official `openai`
Python package pointed at Meta's base URL (https://api.meta.ai/v1). This is the
documented way to call Muse Spark. We never call OpenAI's own servers or models.

Everything the rest of the app sees is ONE normalised shape:
    {"text": str,
     "tool_calls": [{"call_id", "name", "arguments"}],
     "citations": [{"title", "url"}],
     "usage": {"input_tokens", "output_tokens", "reasoning_tokens"},
     "output_items": [...]}   # raw output items, replayed into the next request

Run `python meta_client.py` to send ONE live test request.
"""
import json
import random
import re
import time
import uuid
from datetime import date

import openai
from openai import OpenAI

MAX_RETRIES = 3
REQUEST_TIMEOUT = 120  # seconds; reasoning models can be slow
# Errors that mean "this account cannot use the live API" -> fall back to mock mode.
FALLBACK_KINDS = {"auth", "access", "region"}
REGION_PATTERN = re.compile(r"region|country|geograph|location|not available in your|unsupported", re.I)


class MetaAPIError(Exception):
    """A student-friendly error. `kind` is a short machine-readable category."""

    def __init__(self, message: str, kind: str = "error"):
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------
# Response parsing (shared by the live client and the mock)
# ---------------------------------------------------------------------------

def parse_response(data: dict) -> dict:
    """Turn a raw Responses-API payload into our small normalised dict."""
    if not isinstance(data, dict) or not isinstance(data.get("output"), list):
        raise MetaAPIError("Meta Model API returned an unexpected response shape (no 'output' list).", "bad_shape")

    texts, tool_calls, citations = [], [], []
    for item in data["output"]:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            if not item.get("call_id") or not item.get("name"):
                raise MetaAPIError("Meta Model API returned a tool call without call_id or name.", "bad_shape")
            tool_calls.append({
                "call_id": item["call_id"],
                "name": item["name"],
                "arguments": item.get("arguments") or "{}",
            })
        elif item.get("type") == "message":
            for block in item.get("content") or []:
                if block.get("type") != "output_text":
                    continue
                texts.append(block.get("text") or "")
                for ann in block.get("annotations") or []:
                    if ann.get("type") == "url_citation" and ann.get("url"):
                        citations.append({"title": ann.get("title") or ann["url"], "url": ann["url"]})
        # "reasoning" and "web_search_call" items are kept in output_items but never shown.

    usage = data.get("usage") or {}
    return {
        "text": "\n".join(t for t in texts if t).strip(),
        "tool_calls": tool_calls,
        "citations": list({c["url"]: c for c in citations}.values()),   # de-duplicate by URL
        "usage": {
            "input_tokens": usage.get("input_tokens", 0) or 0,
            "output_tokens": usage.get("output_tokens", 0) or 0,
            "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
        },
        "output_items": [item for item in data["output"] if isinstance(item, dict)],
    }


# ---------------------------------------------------------------------------
# Live client
# ---------------------------------------------------------------------------

class MetaClient:
    """Calls Muse Spark through the Responses API (client.responses.create)."""

    mode = "live"

    def __init__(self, api_key: str, model: str, base_url: str, reasoning_effort: str = "low",
                 http_client=None, sleep=time.sleep):
        if not api_key:
            raise MetaAPIError("MODEL_API_KEY is empty. Add it to your .env file, or set MINI_MUSE_MOCK=1.", "auth")
        self.model = model
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self._secret = api_key
        self._sleep = sleep
        # The SDK does NOT read MODEL_API_KEY by itself, so we pass it explicitly.
        # max_retries=0: we do our own, visible retry loop below.
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT,
                              max_retries=0, http_client=http_client)

    def respond(self, messages, tools=None, use_web_search=False, instructions=None, parallel_tool_calls=True):
        request = {
            "model": self.model,
            "input": messages,
            "reasoning": {"effort": self.reasoning_effort},
            # Keep nothing on Meta's servers; instead ask for encrypted reasoning so we can
            # replay it on the next turn of a multi-step tool loop (documented pattern).
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if instructions:
            request["instructions"] = instructions
        all_tools = list(tools or [])
        if use_web_search:
            all_tools.append({"type": "web_search"})   # built-in tool, Responses API only
        if all_tools:
            request["tools"] = all_tools
            request["parallel_tool_calls"] = parallel_tool_calls

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._client.responses.create(**request)
                return parse_response(response.model_dump(exclude_none=True))
            except (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError) as e:
                last_error = self._friendly(e)
                retry_after = _retry_after_seconds(e)
            except openai.OpenAIError as e:
                raise self._friendly(e) from None
            if attempt < MAX_RETRIES:
                self._sleep(retry_after or (2 ** attempt + random.random()))  # exponential backoff + jitter
        raise last_error

    def _friendly(self, e: Exception) -> MetaAPIError:
        """Map SDK exceptions to short messages students can act on. Never includes the key."""
        status = getattr(e, "status_code", None)
        detail = self._scrub(_api_error_message(e))
        if isinstance(e, openai.APITimeoutError):
            return MetaAPIError("The request to Meta Model API timed out. Try again or lower MUSE_REASONING_EFFORT.", "timeout")
        if isinstance(e, openai.APIConnectionError):
            return MetaAPIError(f"Could not connect to Meta Model API at {self.base_url}. Check your internet connection and MUSE_BASE_URL.", "connection")
        if status in (400, 403, 451) and REGION_PATTERN.search(detail):
            return MetaAPIError(f"Meta Model API is not available for this account's region (HTTP {status}): {detail}", "region")
        if status == 401:
            return MetaAPIError("Meta Model API authentication failed. Check MODEL_API_KEY in your .env file.", "auth")
        if status == 402:
            return MetaAPIError("Meta Model API billing is not set up for this account (HTTP 402). Check the dashboard.", "billing")
        if status == 403:
            return MetaAPIError(f"This API key has no access to model '{self.model}' (HTTP 403). {detail}", "access")
        if status == 404:
            return MetaAPIError(f"Model '{self.model}' was not found (HTTP 404 model_not_found). Check MUSE_MODEL in your .env file.", "model_not_found")
        if status == 429:
            return MetaAPIError("Meta Model API rate limit reached (HTTP 429). Wait a minute and try again.", "rate_limit")
        if status and status >= 500:
            return MetaAPIError(f"Meta Model API server error (HTTP {status}). Try again shortly.", "server")
        if status:
            return MetaAPIError(f"Meta Model API rejected the request (HTTP {status}): {detail}", "bad_request")
        return MetaAPIError(f"Unexpected error talking to Meta Model API ({type(e).__name__}).", "error")

    def _scrub(self, text: str) -> str:
        return text.replace(self._secret, "***") if self._secret else text


def _api_error_message(e) -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        msg = err.get("message") or ""
    else:
        msg = str(body or "")
    return str(msg)[:300]


def _retry_after_seconds(e):
    response = getattr(e, "response", None)
    try:
        value = float(response.headers.get("retry-after")) if response is not None else None
    except (TypeError, ValueError):
        return None
    return min(value, 30) if value else None


# ---------------------------------------------------------------------------
# Mock client: same interface, same response shape, no network.
# ---------------------------------------------------------------------------

class MockMetaClient:
    """A scripted stand-in for Muse Spark.

    It inspects the conversation and the offered tools, then does the "obvious"
    next thing: search -> (request approval via generate_report) -> answer.
    Its responses are built in the Responses-API format and go through the same
    parse_response() as real responses.
    """

    mode = "mock"
    model = "mock-muse-spark"

    def respond(self, messages, tools=None, use_web_search=False, instructions=None, parallel_tool_calls=True):
        tool_names = {t.get("name") for t in tools or [] if t.get("type") == "function"}
        goal = _first_user_text(messages)
        calls = {item["call_id"]: item for item in messages if isinstance(item, dict) and item.get("type") == "function_call"}
        outputs = [(calls[item["call_id"]]["name"], _json_or_text(item.get("output")))
                   for item in messages if isinstance(item, dict) and item.get("type") == "function_call_output"
                   and item.get("call_id") in calls]
        called = {c["name"] for c in calls.values()}

        if not tool_names:
            return _mock_text(
                "(Mock Muse Spark - Basic LLM mode) I have no tools, so I can only answer from what I "
                "learned during training. I cannot search for or verify recent papers, read your PDF, "
                f"or run a calculator. Your question was: \"{goal}\". Switch to Tool-using or Agent mode "
                "to see the model use real tools."
            )

        # Tool-using mode offers no generate_report: after one tool result, just answer.
        if outputs and "generate_report" not in tool_names:
            return _mock_text(_summarise(goal, outputs))

        if "read_pdf" in tool_names and "[A PDF has been uploaded" in goal and "read_pdf" not in called:
            return _mock_call("read_pdf", {"question": goal.split("\n\n[A PDF")[0]})

        question = goal.split("\n\n[A PDF")[0]
        wants_research = bool(RESEARCH_WORDS.search(question))
        expression = _extract_calculation(question, allow_bare=not wants_research)
        if expression and "calculate" in tool_names:
            if "calculate" not in called:
                return _mock_call("calculate", {"expression": expression})
            return _mock_text(_summarise(goal, outputs))

        if wants_research and "search_papers" in tool_names and "search_papers" not in called:
            args = {"query": _extract_topic(goal), "max_results": 5}
            if "recent" in goal.lower():
                args["year_from"] = date.today().year - 2
            return _mock_call("search_papers", args)

        if "generate_report" in tool_names and "generate_report" not in called:
            papers = [p for name, out in outputs if name == "search_papers" and isinstance(out, dict)
                      for p in out.get("papers", [])]
            if papers:
                return _mock_call("generate_report", _mock_report_args(papers))

        if not outputs:                      # a request the script does not understand
            return _mock_text(MOCK_HELP)
        return _mock_text(_summarise(goal, outputs))


def _new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _mock_payload(output):
    return parse_response({
        "id": _new_id("resp_mock"), "object": "response", "model": MockMetaClient.model, "output": output,
        "usage": {"input_tokens": 0, "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}},
    })


def _mock_text(text):
    return _mock_payload([{
        "type": "message", "id": _new_id("msg"), "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }])


def _mock_call(name, arguments):
    return _mock_payload([{
        "type": "function_call", "id": _new_id("fc"), "call_id": _new_id("call"), "name": name,
        "arguments": json.dumps(arguments), "status": "completed",
    }])


def _first_user_text(messages):
    for item in messages:
        if isinstance(item, dict) and item.get("role") == "user":
            content = item.get("content")
            return content if isinstance(content, str) else " ".join(
                b.get("text", "") for b in content or [] if isinstance(b, dict))
    return ""


def _json_or_text(value):
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


RESEARCH_WORDS = re.compile(r"\b(papers?|research|literature|stud(?:y|ies)|articles?|publications?|survey|review)\b", re.I)
NUM = r"(-?\d+(?:\.\d+)?)"

MOCK_HELP = (
    "(Mock Muse Spark) I am a scripted stand-in, so I only understand a few kinds of requests:\n\n"
    "- **Calculations**, e.g. *Calculate 144 / 12*, *Divide 100 by 4*, *What is 15% of 80?*, "
    "*Calculate the percentage increase from 100 to 125.*\n"
    "- **Finding research papers**, e.g. *Find recent papers about battery degradation prediction.*\n"
    "- **Questions about a PDF**: attach one with the + button and ask about it.\n\n"
    "With a real Meta API key, Muse Spark understands any wording."
)


def _extract_calculation(goal, allow_bare=True):
    """Turn simple maths requests into a calculator expression (the mock's only 'understanding')."""
    text = goal.lower().replace("×", "*").replace("÷", "/")
    patterns = [
        (rf"percent(?:age)?\s+(?:increase|decrease|change).*?from\s+{NUM}\s+to\s+{NUM}", "pct_change({0}, {1})"),
        (rf"{NUM}\s*%\s*of\s+{NUM}", "{0} / 100 * {1}"),
        (rf"divide\s+{NUM}\s+by\s+{NUM}", "{0} / {1}"),
        (rf"{NUM}\s+divided\s+by\s+{NUM}", "{0} / {1}"),
        (rf"multiply\s+{NUM}\s+(?:by|and)\s+{NUM}", "{0} * {1}"),
        (rf"{NUM}\s+(?:times|multiplied\s+by)\s+{NUM}", "{0} * {1}"),
        (rf"{NUM}\s+plus\s+{NUM}", "{0} + {1}"),
        (rf"{NUM}\s+minus\s+{NUM}", "{0} - {1}"),
    ]
    for pattern, template in patterns:
        m = re.search(pattern, text)
        if m:
            return template.format(*m.groups())
    if allow_bare:                     # plain arithmetic anywhere, e.g. "what is (3 + 4) * 2?"
        for m in re.finditer(r"[\d\s+\-*/().%^]+", text):
            candidate = m.group().strip().rstrip(".").replace("^", "**")
            if re.search(r"\d\s*(?:\*\*|[+\-*/%])\s*\(?\s*-?\d", candidate):
                return candidate
    return None


def _extract_topic(goal):
    m = re.search(r"(?:papers|research|literature|studies|work)\s+(?:about|on|regarding)\s+(.+?)(?:\s+and\s+|[.?!\n]|$)", goal, re.I)
    return (m.group(1) if m else goal).strip()[:150]


def _mock_report_args(papers):
    first = papers[:5]
    cite_all = [p["id"] for p in first]
    return {
        "summary": f"(Mock) I found {len(papers)} papers and drafted a report outline. Approve to write it.",
        "main_methods": [{"text": f"Retrieved study: \"{p['title']}\" ({p.get('year') or 'n.d.'}).", "sources": [p["id"]]} for p in first[:3]],
        "datasets": [{"text": "Dataset details should be checked in each paper's full text; the mock model does not read full texts.", "sources": []}],
        "models": [{"text": "The retrieved papers are the starting point for comparing model families.", "sources": cite_all}],
        "metrics": [{"text": "Typical forecasting metrics such as RMSE and MAE are likely reported; verify per paper.", "sources": []}],
        "key_findings": [{"text": f"{len(papers)} relevant papers were retrieved from a real academic search API.", "sources": cite_all}],
        "limitations": [{"text": "This outline was produced by the MOCK model from titles only, not by Muse Spark.", "sources": []}],
        "research_gaps": [{"text": "Candidate gap to verify: how well physical constraints generalise across sites and climates.", "sources": []}],
        "future_directions": [{"text": "Run Mini-Muse in live mode to get a real Muse Spark synthesis of these abstracts.", "sources": []}],
    }


def _summarise(goal, outputs):
    lines = ["(Mock Muse Spark) Here is what the tools returned:"]
    for name, out in outputs:
        if isinstance(out, dict) and "error" in out:
            lines.append(f"- {name} failed: {out['error']}")
        elif name == "calculate" and isinstance(out, dict):
            lines.append(f"- The result of {out.get('expression')} is **{out.get('result')}**"
                         + ("%" if "pct_change" in str(out.get("expression")) else "") + ".")
        elif name == "search_papers" and isinstance(out, dict):
            papers = out.get("papers", [])
            lines.append(f"- Found {len(papers)} papers via {out.get('source')}:")
            lines += [f"  [{p['id']}] {p['title']} ({p.get('year') or 'n.d.'})" for p in papers]
        elif name == "read_pdf" and isinstance(out, dict):
            lines.append(f"- Read {len(out.get('excerpts', []))} relevant excerpt(s) from {out.get('file')}.")
        else:
            lines.append(f"- {name} returned a result.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Choosing a client
# ---------------------------------------------------------------------------

class ResilientClient:
    """Uses the live API. If Meta refuses access (bad key / region), switch to mock
    for the rest of the session so the demo keeps working, and remember why."""

    def __init__(self, live=None, mock=None, reason=""):
        self.live = live
        self.mock = mock or MockMetaClient()
        self.mode = "live" if live else "mock"
        self.mock_reason = reason
        self.fallback_error = ""

    def respond(self, *args, **kwargs):
        if self.mode == "live":
            try:
                return self.live.respond(*args, **kwargs)
            except MetaAPIError as e:
                if e.kind not in FALLBACK_KINDS:
                    raise
                self.mode = "mock"
                self.fallback_error = str(e)
                self.mock_reason = "live API refused access"
        return self.mock.respond(*args, **kwargs)


def make_client(settings) -> ResilientClient:
    if settings.force_mock:
        return ResilientClient(reason="MINI_MUSE_MOCK=1")
    if not settings.api_key:
        return ResilientClient(reason="MODEL_API_KEY is empty")
    live = MetaClient(settings.api_key, settings.model, settings.base_url, settings.reasoning_effort)
    return ResilientClient(live=live)


if __name__ == "__main__":
    # Phase 4 smoke test: one live request, no tools.
    import config

    s = config.load_settings()
    if not s.api_key:
        print("MODEL_API_KEY is not set - cannot run a live request. The app will use MOCK MODE.")
        raise SystemExit(1)
    try:
        client = MetaClient(s.api_key, s.model, s.base_url, s.reasoning_effort)
        result = client.respond([{"role": "user", "content": "Reply with exactly: Mini-Muse is connected."}])
        print("LIVE OK:", result["text"])
        print("usage:", result["usage"])
    except MetaAPIError as e:
        print(f"LIVE FAILED [{e.kind}]: {e}")
        raise SystemExit(2)
