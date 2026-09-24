import json

import pytest

import tools

AGENT = tools.MODE_TOOLS["agent"]


def test_schemas_use_responses_api_flat_format():
    for schema in tools.tool_schemas(AGENT):
        assert set(schema) == {"type", "name", "description", "parameters"}
        assert schema["type"] == "function"
        assert schema["parameters"]["type"] == "object"


def test_no_reserved_or_dangerous_tool_names():
    names = set(tools.TOOLS)
    assert not names & {"browser.search", "browser.open", "browser.find"}
    assert names == {"search_papers", "read_pdf", "calculate", "generate_report"}


def test_generate_report_cannot_be_executed_directly():
    assert "generate_report" not in tools.EXECUTORS


def test_basic_mode_has_no_tools():
    assert tools.MODE_TOOLS["basic"] == [] and tools.tool_schemas([]) == []


@pytest.mark.parametrize("name, raw, message", [
    ("run_shell", '{"cmd": "dir"}', "Unknown tool"),
    ("generate_report", "{}", "not available in this mode"),   # tool mode below
    ("calculate", "{not json", "not valid JSON"),
    ("calculate", "[1, 2]", "must be a JSON object"),
    ("calculate", "{}", "expression is required"),
    ("calculate", '{"expression": 5}', "must be of type string"),
    ("calculate", '{"expression": "1+1", "extra": 1}', "not an allowed argument"),
    ("search_papers", '{"query": "solar", "max_results": 50}', "must be <= 10"),
    ("search_papers", '{"query": "solar", "max_results": 0}', "must be >= 1"),
    ("search_papers", '{"query": "solar", "year_from": true}', "must be of type integer"),
    ("search_papers", '{"query": "solar", "year_from": "2024"}', "must be of type integer"),
    ("read_pdf", '{"question": "' + "x" * 600 + '"}', "too long"),
])
def test_invalid_calls_are_rejected(name, raw, message):
    args, error = tools.validate_call(name, raw, tools.MODE_TOOLS["tool"])
    assert args is None and message in error


def test_valid_call_passes():
    args, error = tools.validate_call("search_papers", '{"query": "solar", "year_from": 2024, "max_results": 3}', AGENT)
    assert error is None and args == {"query": "solar", "year_from": 2024, "max_results": 3}


def test_report_claims_are_validated_deeply():
    args = {name: [] for name in tools.REPORT_SECTIONS}
    args["summary"] = "ok"
    args["key_findings"] = [{"text": "claim", "sources": ["one"]}]
    _, error = tools.validate_call("generate_report", json.dumps(args), AGENT)
    assert "key_findings[0].sources[0] must be of type integer" in error


def test_tool_errors_become_observations_not_crashes():
    observation, log = tools.execute("calculate", {"expression": "__import__('os')"}, {"papers": []}, {})
    assert "error" in observation and "failed" in log[0]


def test_read_pdf_without_upload():
    observation, _ = tools.execute("read_pdf", {"question": "what?"}, {"papers": [], "pdf": None}, {})
    assert observation == {"error": "No PDF has been uploaded."}


def test_search_numbers_and_deduplicates_papers(monkeypatch):
    fake = {"source": "Semantic Scholar", "fallback_reason": None, "papers": [
        {"title": "A", "authors": [], "year": 2024, "venue": "", "abstract": "No abstract", "doi": "10.1/a", "url": ""},
        {"title": "B", "authors": [], "year": 2025, "venue": "", "abstract": "No abstract", "doi": "", "url": ""},
    ]}
    captured = {}

    def fake_search(query, year_from, limit, **kwargs):
        captured.update(query=query, year_from=year_from, limit=limit)
        return fake

    monkeypatch.setattr(tools.paper_search, "search_papers", fake_search)
    memory = {"papers": []}
    options = {"max_papers": 3, "year_from": 2023}
    obs, log = tools.execute("search_papers", {"query": "solar", "max_results": 10, "year_from": 2024}, memory, options)
    assert captured == {"query": "solar", "year_from": 2024, "limit": 3}   # sidebar cap wins; later year wins
    assert [p["id"] for p in obs["papers"]] == [1, 2]
    assert log[0] == "Searching Semantic Scholar: 'solar' (2024+)" and log[-1] == "Retrieved 2 papers from Semantic Scholar"
    tools.execute("search_papers", {"query": "solar again"}, memory, options)
    assert len(memory["papers"]) == 2   # same papers again -> same ids, no duplicates
