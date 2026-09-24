"""Paper search tests. All HTTP is faked with httpx.MockTransport - no network."""
import httpx
import pytest

import paper_search
from paper_search import PaperSearchError, search_papers

S2_OK = {"total": 2, "data": [
    {"paperId": "a", "title": "Physics-Informed Solar Forecasting", "authors": [{"name": "Ada Lee"}, {"name": "Bo Kim"}],
     "year": 2024, "venue": "Solar Energy", "abstract": "We forecast PV power.", "externalIds": {"DOI": "10.1/abc"},
     "url": "https://www.semanticscholar.org/paper/a"},
    {"paperId": "b", "title": "Missing Fields Paper", "authors": [], "year": None, "venue": "", "abstract": None,
     "externalIds": None, "url": None},
    {"paperId": "c", "title": None},   # no title -> skipped
]}

OPENALEX_OK = {"results": [{
    "id": "https://openalex.org/W1", "title": "OpenAlex Paper", "publication_year": 2025,
    "doi": "https://doi.org/10.2/xyz",
    "authorships": [{"author": {"display_name": "Cy Park"}}],
    "primary_location": {"source": {"display_name": "Energy Reports"}},
    "abstract_inverted_index": {"Solar": [0], "is": [1], "variable.": [2]},
}]}


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def no_sleep(_seconds):
    pass


def test_semantic_scholar_parsing_and_year_filter():
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=S2_OK)

    result = search_papers("solar", year_from=2024, max_results=5, http=client_for(handler), sleep=no_sleep)
    assert result["source"] == "Semantic Scholar"
    assert seen["params"]["year"] == "2024-"
    assert seen["params"]["limit"] == "5"
    first, second = result["papers"]
    assert first == {"title": "Physics-Informed Solar Forecasting", "authors": ["Ada Lee", "Bo Kim"], "year": 2024,
                     "venue": "Solar Energy", "abstract": "We forecast PV power.", "doi": "10.1/abc",
                     "url": "https://www.semanticscholar.org/paper/a"}
    assert second["abstract"] == "No abstract" and second["doi"] == "" and second["authors"] == []
    assert len(result["papers"]) == 2   # the title-less entry is dropped, never invented


def test_api_key_header_sent():
    def handler(request):
        assert request.headers["x-api-key"] == "k123"
        return httpx.Response(200, json={"data": []})

    search_papers("x y", api_key="k123", http=client_for(handler), sleep=no_sleep)


def test_429_retries_then_falls_back_to_openalex():
    calls = {"s2": 0, "oa": 0}
    sleeps = []

    def handler(request):
        if "semanticscholar" in request.url.host:
            calls["s2"] += 1
            return httpx.Response(429, json={"message": "Too Many Requests"})
        calls["oa"] += 1
        assert request.url.params["filter"] == "from_publication_date:2024-01-01"
        return httpx.Response(200, json=OPENALEX_OK)

    result = search_papers("solar", year_from=2024, http=client_for(handler), sleep=sleeps.append)
    assert calls == {"s2": paper_search.S2_ATTEMPTS, "oa": 1}
    assert len(sleeps) == paper_search.S2_ATTEMPTS - 1
    assert result["source"] == "OpenAlex"
    assert "429" in result["fallback_reason"]
    paper = result["papers"][0]
    assert paper["abstract"] == "Solar is variable."
    assert paper["doi"] == "10.2/xyz" and paper["url"] == "https://doi.org/10.2/xyz"
    assert paper["venue"] == "Energy Reports" and paper["authors"] == ["Cy Park"]


def test_retry_after_header_is_honoured_but_capped():
    sleeps = []

    def handler(request):
        if "semanticscholar" in request.url.host:
            return httpx.Response(429, headers={"Retry-After": "999"})
        return httpx.Response(200, json=OPENALEX_OK)

    search_papers("solar", http=client_for(handler), sleep=sleeps.append)
    assert sleeps and all(s == paper_search.MAX_WAIT_SECONDS for s in sleeps)


def test_s2_recovers_after_one_429():
    responses = iter([httpx.Response(429), httpx.Response(200, json=S2_OK)])
    result = search_papers("solar", http=client_for(lambda r: next(responses)), sleep=no_sleep)
    assert result["source"] == "Semantic Scholar"


def test_timeout_falls_back():
    def handler(request):
        if "semanticscholar" in request.url.host:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=OPENALEX_OK)

    result = search_papers("solar", http=client_for(handler), sleep=no_sleep)
    assert result["source"] == "OpenAlex" and "timed out" in result["fallback_reason"]


def test_both_sources_fail_raises_clear_error():
    with pytest.raises(PaperSearchError, match="Semantic Scholar.*OpenAlex"):
        search_papers("solar", http=client_for(lambda r: httpx.Response(429)), sleep=no_sleep)


def test_no_results_is_not_an_error():
    result = search_papers("zzzz qqqq", http=client_for(lambda r: httpx.Response(200, json={"total": 0})), sleep=no_sleep)
    assert result == {"source": "Semantic Scholar", "papers": [], "fallback_reason": None}


def test_client_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(400) if "semanticscholar" in request.url.host else httpx.Response(200, json=OPENALEX_OK)

    search_papers("solar", http=client_for(handler), sleep=no_sleep)
    assert calls.count("api.semanticscholar.org") == 1
