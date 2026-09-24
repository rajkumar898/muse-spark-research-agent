"""Agent loop tests with MockMetaClient (and a few hand-written fake clients).
Paper search is monkeypatched, so no network is used."""
import json
from pathlib import Path

import pytest

import agent
import tools
from config import MAX_STEPS
from meta_client import MetaAPIError, MockMetaClient, _mock_call, _mock_text

GOAL_A = "Find recent papers about physics-informed solar power forecasting and identify the main research gaps."
GOAL_B = "Calculate the percentage increase from 100 to 125."

FAKE_PAPERS = [
    {"title": f"Paper {i}", "authors": ["A. Author"], "year": 2025, "venue": "Solar Energy",
     "abstract": "Abstract text.", "doi": f"10.1/{i}", "url": f"https://doi.org/10.1/{i}"}
    for i in range(1, 4)
]


@pytest.fixture(autouse=True)
def fake_search(monkeypatch):
    calls = []

    def search(query, year_from, limit, **kwargs):
        calls.append({"query": query, "year_from": year_from, "limit": limit})
        return {"source": "Semantic Scholar", "papers": [dict(p) for p in FAKE_PAPERS[:limit]], "fallback_reason": None}

    monkeypatch.setattr(tools.paper_search, "search_papers", search)
    return calls


@pytest.fixture
def options(tmp_path):
    return {"max_papers": 5, "year_from": None, "use_web_search": False, "model": "mock", "outputs_dir": str(tmp_path)}


class CountingClient:
    """Wraps a client and counts calls, so we can prove approval never re-calls the model."""

    def __init__(self, inner):
        self.inner, self.calls, self.last_tools = inner, 0, None

    def respond(self, messages, tools=None, **kwargs):
        self.calls += 1
        self.last_tools = tools
        return self.inner.respond(messages, tools=tools, **kwargs)


def run(goal, mode, options, client=None):
    state = agent.new_state()
    client = client or CountingClient(MockMetaClient())
    agent.start(state, goal, mode, options, client)
    return state, client


def test_agent_pauses_for_approval_and_does_not_write_report(options, tmp_path, fake_search):
    state, client = run(GOAL_A, "agent", options)
    assert state["status"] == agent.WAITING_APPROVAL
    assert state["pending_report"] is not None
    assert fake_search[0]["query"] == "physics-informed solar power forecasting"
    assert [p["id"] for p in state["papers"]] == [1, 2, 3]
    assert list(tmp_path.iterdir()) == []                    # nothing written before approval
    assert state["log"][-1] == "Waiting for your approval"
    assert client.calls == 2                                 # search decision + report decision


def test_rerun_while_waiting_does_nothing(options):
    state, client = run(GOAL_A, "agent", options)
    snapshot = json.dumps(state, sort_keys=True)
    agent.approve({**state, "status": agent.RUNNING})        # approve in the wrong state is ignored
    assert json.dumps(state, sort_keys=True) == snapshot
    assert client.calls == 2


def test_approve_resumes_and_writes_report(options, tmp_path):
    state, client = run(GOAL_A, "agent", options)
    agent.approve(state)
    assert state["status"] == agent.DONE
    assert client.calls == 2                                 # approval did NOT call the model again
    report = Path(state["report_path"])
    assert report.parent == tmp_path and report.exists()
    text = report.read_text(encoding="utf-8")
    for heading in ["Research question", "Sources", "Main methods", "Datasets", "Models", "Metrics",
                    "Key findings", "Limitations", "Research gaps", "Future directions"]:
        assert f"## {heading}" in text
    assert "[1] A. Author (2025). *Paper 1*." in text
    assert "*(AI synthesis)*" in text and "[1][2][3]" in text
    assert state["history"][-1]["type"] == "function_call_output"
    agent.approve(state)                                     # a second click is a no-op
    assert len(list(tmp_path.iterdir())) == 1


def test_reject_stops_cleanly(options, tmp_path):
    state, client = run(GOAL_A, "agent", options)
    agent.reject(state)
    assert state["status"] == agent.DONE
    assert state["pending_report"] is None and state["report_path"] == ""
    assert "rejected" in state["final_answer"]
    assert list(tmp_path.iterdir()) == []
    assert "rejected" in json.loads(state["history"][-1]["output"])["error"]
    assert client.calls == 2


def test_calculation_goal_in_agent_mode(options):
    state, _ = run(GOAL_B, "agent", options)
    assert state["status"] == agent.DONE
    assert state["tool_results"][0]["observation"] == {"expression": "pct_change(100, 125)", "result": 25.0}
    assert "25.0" in state["final_answer"] and "%" in state["final_answer"]


def test_basic_mode_sends_no_tools(options, fake_search):
    state, client = run(GOAL_A, "basic", options)
    assert state["status"] == agent.DONE and client.last_tools is None
    assert fake_search == [] and state["papers"] == []


def test_tool_mode_uses_one_tool_then_answers(options, fake_search):
    state, client = run(GOAL_A, "tool", options)
    assert state["status"] == agent.DONE
    assert len(fake_search) == 1 and client.calls == 2
    assert state["pending_report"] is None and "Paper 1" in state["final_answer"]


class GreedyClient:
    """A misbehaving model that never stops calling tools."""

    def __init__(self):
        self.calls = 0

    def respond(self, messages, **kwargs):
        self.calls += 1
        return _mock_call("calculate", {"expression": f"{self.calls} + 1"})


def test_agent_respects_max_steps(options):
    client = GreedyClient()
    state, _ = run(GOAL_B, "agent", options, client)
    assert state["status"] == agent.DONE
    assert state["steps"] == MAX_STEPS and client.calls == MAX_STEPS
    assert f"{MAX_STEPS} tool rounds" in state["final_answer"]


def test_tool_mode_allows_only_one_round(options):
    state, _ = run(GOAL_B, "tool", options, GreedyClient())
    assert state["status"] == agent.DONE and len(state["tool_results"]) == 1
    assert "only one tool round" in state["final_answer"]


class ScriptedClient:
    def __init__(self, *responses):
        self.responses = list(responses)

    def respond(self, messages, **kwargs):
        return self.responses.pop(0)


def test_unknown_tool_becomes_error_observation(options):
    client = ScriptedClient(_mock_call("delete_files", {"path": "/"}), _mock_text("OK, I cannot do that."))
    state, _ = run(GOAL_B, "agent", options, client)
    assert state["status"] == agent.DONE
    assert "Unknown tool" in state["tool_results"][0]["observation"]["error"]
    assert state["final_answer"] == "OK, I cannot do that."


def test_report_before_search_is_refused(options):
    report_args = {name: [] for name in tools.REPORT_SECTIONS} | {"summary": "done"}
    client = ScriptedClient(_mock_call("generate_report", report_args), _mock_text("Understood."))
    state, _ = run(GOAL_A, "agent", options, client)
    assert state["status"] == agent.DONE and state["pending_report"] is None
    assert "Search for papers first" in json.loads(state["history"][2]["output"])["error"]


def test_invalid_report_arguments_are_refused(options):
    client = ScriptedClient(_mock_call("search_papers", {"query": "solar"}),
                            _mock_call("generate_report", {"summary": "missing sections"}),
                            _mock_text("I will stop here."))
    state, _ = run(GOAL_A, "agent", options, client)
    assert state["status"] == agent.DONE and state["pending_report"] is None


def test_api_error_sets_error_state(options):
    class Broken:
        def respond(self, *args, **kwargs):
            raise MetaAPIError("Meta Model API rate limit reached (HTTP 429).", "rate_limit")

    state, _ = run(GOAL_A, "agent", options, Broken())
    assert state["status"] == agent.ERROR and "429" in state["error"]


def test_pdf_is_offered_and_read(options):
    state = agent.new_state(pdf={"name": "p.pdf", "pages": 1, "words": 5, "chunks": ["solar physics text"]})
    agent.start(state, GOAL_A, "agent", options, MockMetaClient())
    assert "[A PDF has been uploaded: p.pdf" in state["history"][0]["content"]
    assert state["tool_results"][0]["tool"] == "read_pdf"
    assert state["tool_results"][0]["observation"]["excerpts"][0]["text"] == "solar physics text"


def test_start_keeps_pdf_but_clears_the_rest(options):
    state, _ = run(GOAL_A, "agent", options)
    state["pdf"] = {"name": "keep.pdf", "pages": 1, "words": 1, "chunks": ["x"]}
    agent.start(state, GOAL_B, "basic", options, MockMetaClient())
    assert state["pdf"]["name"] == "keep.pdf" and state["papers"] == [] and state["goal"] == GOAL_B
