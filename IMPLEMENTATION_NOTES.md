# Implementation Notes: Mini-Muse

**API facts verified:** 24 September 2026, by reading the official docs at https://dev.meta.ai/docs/*
(overview, quickstart, authentication, models, tool-calling, search-grounding, error-handling,
pricing-rate-limits, reasoning, protocols/responses).

## 1. Live vs mock status

| Item | Result (24 Sep 2026) |
|---|---|
| `MODEL_API_KEY` available? | **No.** It was not set in the environment and there is no `.env`. |
| Phase 4 live request | **Not possible without a key.** `python meta_client.py` → "MODEL_API_KEY is not set". |
| Endpoint reachability | `POST https://api.meta.ai/v1/responses` with no key → **HTTP 401** `{"error":{"code":"invalid_api_key","message":"Unauthorized","param":null,"type":"authentication_error"}}`. This matches the documented error shape. |
| Client error path against the real server | `MetaClient` with a dummy key → real 401 → "Meta Model API authentication failed. Check MODEL_API_KEY in your .env file." `ResilientClient` then switched to mock mode, as designed. |
| Region restriction | Not observed (we got a normal 401, not a geo refusal). The docs we read say nothing about region limits. We cannot confirm whether a real key from this region would be accepted. |
| End-to-end demos | Ran in **MOCK MODE** (model scripted). **Paper search was live** (real Semantic Scholar/OpenAlex). |

## 2. Verified API facts

| Fact | Verified value | Source page |
|---|---|---|
| Base URL | `https://api.meta.ai/v1` | quickstart, overview |
| Endpoint used | `POST /v1/responses` (Responses API) | protocols/responses |
| Model IDs | `muse-spark-1.3` (default, latest), `muse-spark-1.2`, `muse-spark-1.1`, plus `-contributor` variants of 1.3 and 1.2 (cheaper; prompts may be used for training) | models |
| Context window | 1,048,576 tokens | models |
| Auth | `Authorization: Bearer <key>`; env var `MODEL_API_KEY` | authentication |
| SDK | Official `openai` Python SDK with `base_url` set to Meta. It does **not** read `MODEL_API_KEY` automatically, so we pass it explicitly | quickstart |
| Reasoning effort | Responses API: `reasoning={"effort": ...}` (Chat Completions: `reasoning_effort`). Values: minimal, low, medium, high, xhigh, max. `none` → HTTP 400 on Muse Spark | reasoning |
| Multi-turn reasoning | Use `previous_response_id`, or `include=["reasoning.encrypted_content"]` and replay the reasoning items | reasoning, protocols/responses |
| Function tool shape (Responses) | Flat: `{"type":"function","name","description","parameters"}` | tool-calling |
| Tool call output item | `{"type":"function_call","call_id","name","arguments":"<json string>","status"}` | tool-calling |
| Tool result input item | `{"type":"function_call_output","call_id","output":"<string>"}` | tool-calling |
| `tool_choice` | Only `"auto"` is supported; anything else → 400 | tool-calling |
| `parallel_tool_calls` | Default `true` | tool-calling |
| Function names | Letters, digits, `_`, `-` and at most one `.` (two or more dots → 400) | tool-calling |
| `call_id` | 1–64 characters | tool-calling |
| Web search | `{"type":"web_search"}`, **Responses API only**. Citations come as `url_citation` annotations (`url`, `title`, `start_index`, `end_index`). Reserved names: `browser.search`, `browser.open`, `browser.find`. $2.50 per 1,000 searches | search-grounding, pricing |
| Errors | 400 invalid_request_error · 401 authentication_error/invalid_api_key · 402 billing_error · 403 permission · 404 model_not_found · 413 · 429 rate_limit_exceeded · 500/503/504. Body: `{"error":{"message","type","param","code"}}`. Retry only 429/5xx, with exponential backoff and jitter, honouring `Retry-After` | error-handling |
| Pricing (standard) | Input $1.25/M, cached input $0.15/M, output $4.25/M. `output_tokens` includes reasoning tokens (`usage.output_tokens_details.reasoning_tokens`) | pricing, protocols/responses |
| Rate limits | Standard: 3,000 RPM, 4M TPM per team | pricing |

**Differences from the notes in the task brief:** none that matter.
- The brief says reasoning effort should be low/medium "if docs show how". They do show it, as `reasoning.effort`, and we default to `low`.
- The quickstart page only lists 1.3 and 1.1, but the Models page lists 1.2 as well.
- "Public preview, US-only" is **not** stated anywhere in the pages we read, so we could not verify it.

## 3. Request format we send

```python
client = OpenAI(api_key=MODEL_API_KEY, base_url="https://api.meta.ai/v1", max_retries=0, timeout=120)
client.responses.create(
    model="muse-spark-1.3",
    instructions=prompts.AGENT,                 # per-mode system instructions
    input=history,                              # list of Responses-API items (see below)
    tools=[{"type": "function", "name": "search_papers", "description": "...", "parameters": {...}}, ...]
          (+ {"type": "web_search"} if the toggle is on),
    parallel_tool_calls=True,                   # False in Tool-using mode
    reasoning={"effort": "low"},
    store=False,
    include=["reasoning.encrypted_content"],
)
```

`history` starts as `[{"role": "user", "content": goal}]`. After each call we append **all** output
items (reasoning with encrypted content, function_call, message), then one
`function_call_output` per executed call. This is the documented manual-history pattern. We use it
instead of `previous_response_id` so the whole conversation lives in `st.session_state` and survives
Streamlit reruns. `store=False` means nothing is stored server-side.

*Unverified live:* whether Meta accepts replayed items with every field produced by
`model_dump(exclude_none=True)`. It follows the documented pattern, but nobody has run it with a real key yet.

## 4. Agent architecture

- `agent.py` holds one JSON-friendly `state` dict: status, mode, goal, history, log, papers, pdf,
  tool_results, pending_report, final_answer, report, usage, steps.
- **Modes:** basic (no tools) · tool (one tool round, `parallel_tool_calls=False`) · agent (loop).
- **Hard limit:** `MAX_STEPS = 8` tool rounds. After that the agent stops with a message.
- **Approval:** `generate_report` is not in `tools.EXECUTORS`, so the only way it runs is through
  `agent.approve()`, which is triggered by the Approve button. Approve and Reject never call the model.
- **Model calls** happen only inside `agent.start()`, which runs only from the Start button.
- **Normalised model result:** `{text, tool_calls[{call_id,name,arguments}], citations[{title,url}], usage, output_items}`.
  `MockMetaClient` builds raw Responses-API payloads and passes them through the same `parse_response()`.
- **Fallback:** `ResilientClient` switches to mock on error kinds `auth`, `access` and `region`.
  Other errors (429 after retries, 5xx, timeout, 400, 404) are shown to the user as an `ERROR` state.

## 5. External APIs

| API | Endpoint | Notes |
|---|---|---|
| Meta Model API | `POST https://api.meta.ai/v1/responses` | Bearer `MODEL_API_KEY`. We retry 429/5xx/timeouts 3×; the SDK's own retries are off. |
| Semantic Scholar | `GET https://api.semanticscholar.org/graph/v1/paper/search?query&limit&fields=title,authors,year,venue,abstract,externalIds,url&year=YYYY-` | Optional `x-api-key`. 3 attempts (2 s, 4 s backoff or `Retry-After`, capped at 40 s). |
| OpenAlex | `GET https://api.openalex.org/works?search&per-page&filter=from_publication_date:YYYY-01-01` | Optional `api_key`. 2 attempts honouring `Retry-After` (≤ 40 s). The abstract is rebuilt from `abstract_inverted_index`. |

**Observed on 24 Sep 2026:** anonymous Semantic Scholar returned HTTP 429 on most requests.
Anonymous OpenAlex returned 429 with the message "Anonymous search is temporarily rate-limited
… Retry-After: 37". Both recovered after waiting, so rate limiting is the main real-world risk
for live demos. Free API keys and the 1-hour in-process cache reduce it.

## 6. Dependencies (installed in `.venv`, Python 3.14.6)

streamlit 1.64.0 · openai 3.19.2 · python-dotenv 1.2.3 · httpx 0.28.1 · pypdf 6.19.0 · pytest 9.1.1

## 7. Test results (24 Sep 2026)

```
pytest -q        → 126 passed, 2 skipped (live tests; RUN_LIVE_TESTS not set) in 0.8 s
```

**Final validation (mock model, live paper search):**
- **A.** Physics-informed solar forecasting. Semantic Scholar returned 429 ×3, so the search fell back to OpenAlex and got 5 real papers with DOIs (e.g. 10.1186/s42162-025-00604-7, 10.1016/j.egyr.2026.109068). The agent reached `WAITING_APPROVAL`; Approve → report saved to `outputs/report_20260924_101035_…md` → `DONE`. It took 44 s, mostly rate-limit waits.
- **B.** "Calculate the percentage increase from 100 to 125" → `calculate("pct_change(100, 125)")` = **25.0%**. No approval was needed → `DONE`.
- **C.** Reject path.
  - First try: both search APIs returned 429 on an immediate repeat. The agent received the error observation and stopped cleanly, without crashing, but it never reached the approval step.
  - Retried after adding the cache: Semantic Scholar returned 5 papers → `WAITING_APPROVAL` → Reject → `DONE`, with no report file written. A second run was served from the cache in under 1 s.
- **UI (Streamlit `AppTest`, mock model, stubbed search):**
  - Start → `WAITING_APPROVAL` after 2 model calls. Two plain reruns made **no extra model calls**.
  - Approve → `DONE` with a report; Reject → `DONE` without one. No exceptions.
  - The MOCK MODE badge is visible.
  - All three modes give visibly different answers to the calculator question.
- **Headless launch:** `streamlit run app.py --server.headless true` → `/_stcore/health` = `ok`, with no errors in the log.

## 8. Known limitations

See README → Limitations. The key ones:
- **No live Muse Spark run yet**, because there was no key.
- Anonymous search APIs rate-limit heavily.
- PDF relevance is keyword-based.
- `web_search` is untested for this account.

## 9. Reproducibility

```bash
cd muse-spark-research-agent
python -m venv .venv && .venv\Scripts\activate         # Windows
pip install -r requirements.txt
copy .env.example .env                                   # add MODEL_API_KEY (or set MINI_MUSE_MOCK=1)
python meta_client.py                                    # one live request (Phase 4 check)
pytest -q                                                # offline test suite
RUN_LIVE_TESTS=1 pytest tests/test_live.py               # optional live tests
streamlit run app.py
```
