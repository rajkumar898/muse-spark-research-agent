"""The agent loop, written as plain Python so every step is visible.

State machine (the whole state is one JSON-friendly dict, kept in st.session_state):

    IDLE -> RUNNING -> WAITING_APPROVAL -> APPROVED -> REPORTING -> DONE
                    |                   -> REJECTED -> DONE
                    -> DONE (final answer, or MAX_STEPS reached)
                    -> ERROR

The model is only ever called from start/run (the "Start" button). Approve and
Reject never call the model, so a Streamlit rerun can never repeat an API call.
Human approval is enforced HERE, in code: generate_report is never executed
directly when the model asks for it - we pause and wait for a human.
"""
import json
from pathlib import Path

import prompts
import report_generator
import tools
from config import MAX_STEPS, OUTPUTS_DIR
from meta_client import MetaAPIError

IDLE, RUNNING, WAITING_APPROVAL = "IDLE", "RUNNING", "WAITING_APPROVAL"
APPROVED, REJECTED, REPORTING = "APPROVED", "REJECTED", "REPORTING"
DONE, ERROR = "DONE", "ERROR"

MAX_OBSERVATION_CHARS = 60_000   # keep one tool result from flooding the context
MODE_LABELS = {"basic": "Basic LLM", "tool": "Tool-using LLM", "agent": "Agent"}


def new_state(pdf=None) -> dict:
    """Fresh session memory. Only the uploaded PDF may be carried over."""
    return {
        "status": IDLE,
        "mode": "agent",
        "goal": "",
        "options": {},
        "history": [],          # the conversation sent to Muse Spark (Responses API input items)
        "log": [],              # short human-readable action lines (never model reasoning)
        "papers": [],           # every paper retrieved so far, numbered 1..n
        "pdf": pdf,             # {"name", "pages", "words", "chunks"} or None
        "tool_results": [],     # [{"tool", "arguments", "observation"}]
        "citations": [],        # web-search citations, if web search is on
        "steps": 0,             # tool rounds used
        "pending_report": None, # {"call_id", "arguments"} while waiting for approval
        "final_answer": "",
        "report_markdown": "",
        "report_path": "",
        "error": "",
        "usage": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
    }


def _log(state, line):
    state["log"].append(line)
    if state.get("_on_log"):          # optional live-progress callback from the UI
        state["_on_log"](line)


def _set_status(state, status):
    state["status"] = status


# ---------------------------------------------------------------------------
# Public API used by app.py
# ---------------------------------------------------------------------------

def start(state, goal, mode, options, client, on_log=None):
    """Begin a new run for `goal`, then advance until done or paused.
    `on_log(line)` is called for each activity line so the UI can show progress live."""
    pdf = state.get("pdf")
    state.clear()
    state.update(new_state(pdf=pdf))
    state.update(goal=goal.strip(), mode=mode, options={"outputs_dir": str(OUTPUTS_DIR), **options})

    user_text = state["goal"]
    if pdf and mode != "basic":
        user_text += f"\n\n[A PDF has been uploaded: {pdf['name']} ({pdf['pages']} pages). Use read_pdf to consult it.]"
    state["history"] = [{"role": "user", "content": user_text}]
    state["_on_log"] = on_log
    _set_status(state, RUNNING)
    _log(state, f"Goal received ({MODE_LABELS[mode]} mode)")

    try:
        if mode == "basic":
            _run_basic(state, client)
        elif mode == "tool":
            _run_tool_once(state, client)
        else:
            _run_agent_loop(state, client)
    except MetaAPIError as e:
        state["error"] = str(e)
        _set_status(state, ERROR)
        _log(state, f"Stopped with an error: {e}")
    finally:
        state.pop("_on_log", None)    # keep the saved state plain data


def approve(state):
    """Human pressed Approve: now (and only now) the report tool runs."""
    if state["status"] != WAITING_APPROVAL or not state["pending_report"]:
        return
    _set_status(state, APPROVED)
    _log(state, "You approved the report")
    _set_status(state, REPORTING)
    _log(state, "Writing the research report")

    pending = state["pending_report"]
    markdown = report_generator.build_report(
        state["goal"], state["papers"], pending["arguments"],
        model=state["options"].get("model", "muse-spark"), web_citations=state["citations"],
    )
    path = report_generator.save_report(markdown, Path(state["options"]["outputs_dir"]), state["goal"])
    _add_observation(state, pending["call_id"], {"status": "report written", "file": path.name})

    state.update(pending_report=None, report_markdown=markdown, report_path=str(path), final_answer=markdown)
    _log(state, f"Report saved to outputs/{path.name}")
    _set_status(state, DONE)


def reject(state):
    """Human pressed Reject: the report tool never runs, the agent stops cleanly."""
    if state["status"] != WAITING_APPROVAL or not state["pending_report"]:
        return
    _set_status(state, REJECTED)
    _add_observation(state, state["pending_report"]["call_id"], {"error": "The human rejected the report."})
    state["pending_report"] = None
    state["final_answer"] = "Stopped: you rejected the report, so it was not written. The sources found are listed above."
    _log(state, "You rejected the report. The agent stopped without writing it")
    _set_status(state, DONE)


# ---------------------------------------------------------------------------
# The three demo modes
# ---------------------------------------------------------------------------

def _ask_model(state, client, allowed_tools, parallel=True):
    """One call to Muse Spark. Its output items are appended to the conversation."""
    result = client.respond(
        state["history"],
        tools=tools.tool_schemas(allowed_tools) or None,
        use_web_search=state["options"].get("use_web_search", False) and state["mode"] != "basic",
        instructions=prompts.BY_MODE[state["mode"]],
        parallel_tool_calls=parallel,
    )
    state["history"].extend(result["output_items"])
    for key in state["usage"]:
        state["usage"][key] += result["usage"].get(key, 0)
    state["citations"] += [c for c in result["citations"] if c not in state["citations"]]
    return result


def _run_basic(state, client):
    """Mode 1: user -> Muse Spark -> answer. No tools at all."""
    _log(state, "Asking Muse Spark (no tools)")
    result = _ask_model(state, client, allowed_tools=[])
    _finish(state, result["text"])


def _run_tool_once(state, client):
    """Mode 2: user -> Muse Spark -> (at most) one tool round -> Muse Spark -> answer."""
    allowed = tools.MODE_TOOLS["tool"]
    _log(state, "Asking Muse Spark (tools available, one round allowed)")
    result = _ask_model(state, client, allowed, parallel=False)
    if not result["tool_calls"]:
        return _finish(state, result["text"])

    for call in result["tool_calls"]:
        _run_tool_call(state, call, allowed)
    _log(state, "Sending the tool result back to Muse Spark")
    result = _ask_model(state, client, allowed, parallel=False)
    if result["tool_calls"]:
        wanted = ", ".join(c["name"] for c in result["tool_calls"])
        note = (f"\n\n*Tool-using mode allows only one tool round. Muse Spark wanted to call {wanted} next "
                "- switch to Agent mode to let it keep going.*")
        _log(state, "Stopped: one tool round used (Muse Spark wanted another)")
        return _finish(state, (result["text"] or "No final answer yet.") + note)
    _finish(state, result["text"])


def _run_agent_loop(state, client):
    """Mode 3: decide -> act -> observe -> decide ... until an answer or approval."""
    allowed = tools.MODE_TOOLS["agent"]
    while state["status"] == RUNNING:
        if state["steps"] >= MAX_STEPS:
            _log(state, f"Stopped: reached the limit of {MAX_STEPS} tool rounds")
            return _finish(state, f"Stopped after {MAX_STEPS} tool rounds without a final answer. "
                                  "Try a narrower question.")

        _log(state, f"Step {state['steps'] + 1}: asking Muse Spark what to do next")
        result = _ask_model(state, client, allowed)

        if not result["tool_calls"]:            # plain text = final answer
            return _finish(state, result["text"])

        state["steps"] += 1
        for call in result["tool_calls"]:
            if call["name"] == "generate_report":
                _request_approval(state, call, allowed)
            else:
                _run_tool_call(state, call, allowed)

        if state["pending_report"]:
            _set_status(state, WAITING_APPROVAL)
            _log(state, "Waiting for your approval")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_tool_call(state, call, allowed):
    _log(state, f"Muse Spark chose tool: {call['name']}")
    args, error = tools.validate_call(call["name"], call["arguments"], allowed)
    if error:
        observation = {"error": error}
        _log(state, f"Tool call refused by validator: {error}")
    else:
        observation, lines = tools.execute(call["name"], args, state, state["options"])
        for line in lines:
            _log(state, line)
    state["tool_results"].append({"tool": call["name"], "arguments": args, "observation": observation})
    _add_observation(state, call["call_id"], observation)


def _request_approval(state, call, allowed):
    """The model wants the report. We validate it but do NOT run it - we pause instead."""
    _log(state, "Muse Spark wants to write the report")
    args, error = tools.validate_call(call["name"], call["arguments"], allowed)
    if not error and not state["papers"]:
        error = "Search for papers first: a report needs retrieved sources."
    if not error and state["pending_report"]:
        error = "A report is already waiting for approval."
    if error:
        _log(state, f"Report request refused by validator: {error}")
        return _add_observation(state, call["call_id"], {"error": error})
    state["pending_report"] = {"call_id": call["call_id"], "arguments": args}


def _add_observation(state, call_id, observation):
    text = json.dumps(observation, ensure_ascii=False)
    if len(text) > MAX_OBSERVATION_CHARS:
        text = text[:MAX_OBSERVATION_CHARS] + "... [truncated]"
    state["history"].append({"type": "function_call_output", "call_id": call_id, "output": text})


def _finish(state, text):
    state["final_answer"] = text or "(Muse Spark returned an empty answer.)"
    _log(state, "Final answer ready")
    _set_status(state, DONE)
