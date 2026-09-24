"""Academic paper search tool.

Primary source: Semantic Scholar Graph API (/paper/search).
Fallback: OpenAlex (/works) if Semantic Scholar keeps failing (e.g. HTTP 429).

We only ever return what the APIs return. Nothing is invented.
"""
import time

import httpx

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,authors,year,venue,abstract,externalIds,url"
OPENALEX_URL = "https://api.openalex.org/works"
S2_ATTEMPTS = 3
OPENALEX_ATTEMPTS = 2
MAX_WAIT_SECONDS = 40     # never wait longer than this for a Retry-After
TIMEOUT_SECONDS = 20


class PaperSearchError(Exception):
    pass


CACHE_SECONDS = 3600
_cache = {}   # (query, year_from, max_results) -> (time, result). Repeated demo queries skip the rate limits.


def search_papers(query, year_from=None, max_results=5, *, api_key="", openalex_key="", http=None, sleep=time.sleep):
    """Return {"source": ..., "papers": [...], "fallback_reason": ...}.

    `http` and `sleep` can be replaced in tests so no real network call is made
    (the cache is only used for real network calls).
    """
    key = (query.strip().lower(), year_from, max_results)
    if http is None and key in _cache and time.time() - _cache[key][0] < CACHE_SECONDS:
        return {**_cache[key][1], "cached": True}

    owns_client = http is None
    http = http or httpx.Client(timeout=TIMEOUT_SECONDS, headers={"User-Agent": "Mini-Muse/1.0 (educational)"})
    try:
        try:
            papers = _search_semantic_scholar(http, query, year_from, max_results, api_key, sleep)
            result = {"source": "Semantic Scholar", "papers": papers, "fallback_reason": None}
        except PaperSearchError as s2_error:
            try:
                papers = _search_openalex(http, query, year_from, max_results, openalex_key, sleep)
            except PaperSearchError as oa_error:
                raise PaperSearchError(
                    f"Paper search failed. Semantic Scholar: {s2_error}. OpenAlex: {oa_error}."
                ) from None
            result = {"source": "OpenAlex", "papers": papers, "fallback_reason": str(s2_error)}
    finally:
        if owns_client:
            http.close()
    if owns_client:
        _cache[key] = (time.time(), result)
    return result


# ---------- Semantic Scholar ----------

def _search_semantic_scholar(http, query, year_from, max_results, api_key, sleep):
    params = {"query": query, "limit": max_results, "fields": S2_FIELDS}
    if year_from:
        params["year"] = f"{year_from}-"          # "2024-" means 2024 or later
    headers = {"x-api-key": api_key} if api_key else {}

    problem = "unknown error"
    for attempt in range(S2_ATTEMPTS):
        resp = None
        try:
            resp = http.get(S2_URL, params=params, headers=headers)
        except httpx.TimeoutException:
            problem = "request timed out"
        except httpx.HTTPError as e:
            problem = f"network error ({type(e).__name__})"
        else:
            if resp.status_code == 200:
                try:
                    items = resp.json().get("data") or []
                except ValueError:
                    raise PaperSearchError("invalid JSON response") from None
                return [p for p in (_from_s2(item) for item in items) if p]
            if resp.status_code != 429 and resp.status_code < 500:
                raise PaperSearchError(f"HTTP {resp.status_code}")
            problem = "rate-limited (HTTP 429)" if resp.status_code == 429 else f"server error (HTTP {resp.status_code})"
        if attempt < S2_ATTEMPTS - 1:
            sleep(_wait_seconds(resp, default=2 ** (attempt + 1)))  # back off: 2s, 4s (or Retry-After)
    raise PaperSearchError(f"{problem} after {S2_ATTEMPTS} attempts")


def _from_s2(item):
    if not isinstance(item, dict) or not item.get("title"):
        return None
    doi = (item.get("externalIds") or {}).get("DOI") or ""
    return {
        "title": item["title"].strip(),
        "authors": [a.get("name", "") for a in item.get("authors") or [] if a.get("name")],
        "year": item.get("year"),
        "venue": item.get("venue") or "",
        "abstract": item.get("abstract") or "No abstract",
        "doi": doi,
        "url": item.get("url") or (f"https://doi.org/{doi}" if doi else ""),
    }


# ---------- OpenAlex ----------

def _search_openalex(http, query, year_from, max_results, api_key, sleep):
    params = {"search": query, "per-page": max_results}
    if year_from:
        params["filter"] = f"from_publication_date:{year_from}-01-01"
    if api_key:
        params["api_key"] = api_key
    for attempt in range(OPENALEX_ATTEMPTS):
        try:
            resp = http.get(OPENALEX_URL, params=params)
        except httpx.TimeoutException:
            raise PaperSearchError("request timed out") from None
        except httpx.HTTPError as e:
            raise PaperSearchError(f"network error ({type(e).__name__})") from None
        if resp.status_code != 429 or attempt == OPENALEX_ATTEMPTS - 1:
            break
        sleep(_wait_seconds(resp, default=2))
    if resp.status_code == 429:
        raise PaperSearchError("rate-limited (HTTP 429). A free OPENALEX_API_KEY avoids this: https://openalex.org/rest-api")
    if resp.status_code != 200:
        raise PaperSearchError(f"HTTP {resp.status_code}")
    try:
        items = resp.json().get("results") or []
    except ValueError:
        raise PaperSearchError("invalid JSON response") from None
    return [p for p in (_from_openalex(item) for item in items) if p][:max_results]


def _from_openalex(item):
    title = (item or {}).get("title") or (item or {}).get("display_name")
    if not title:
        return None
    doi = (item.get("doi") or "").replace("https://doi.org/", "")
    source = ((item.get("primary_location") or {}).get("source") or {})
    return {
        "title": title.strip(),
        "authors": [
            (a.get("author") or {}).get("display_name", "")
            for a in item.get("authorships") or []
            if (a.get("author") or {}).get("display_name")
        ],
        "year": item.get("publication_year"),
        "venue": source.get("display_name") or "",
        "abstract": _rebuild_abstract(item.get("abstract_inverted_index")) or "No abstract",
        "doi": doi,
        "url": f"https://doi.org/{doi}" if doi else item.get("id", ""),
    }


def _wait_seconds(resp, default):
    """Honour the server's Retry-After header, capped so the UI never hangs for long."""
    try:
        value = float(resp.headers.get("retry-after")) if resp is not None else default
    except (TypeError, ValueError):
        value = default
    return min(max(value, 0), MAX_WAIT_SECONDS)


def _rebuild_abstract(inverted_index):
    """OpenAlex stores abstracts as {word: [positions]}; turn it back into text."""
    if not isinstance(inverted_index, dict):
        return ""
    positions = [(pos, word) for word, pos_list in inverted_index.items() for pos in pos_list]
    return " ".join(word for _, word in sorted(positions))
