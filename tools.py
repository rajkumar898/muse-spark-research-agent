"""Tool registry: the ONLY actions Muse Spark can trigger.

Each tool has a JSON Schema that is sent to the model (native function calling).
Before anything runs, our Python code checks:
  1. the tool name is registered and allowed in the current mode, and
  2. the arguments match the schema.
If either check fails, the model gets an error observation instead - nothing crashes.

There is deliberately no shell, Python, file-system or environment-variable tool.
"""
import json

import calculator
import paper_search
import pdf_reader

# A "claim" in the report: a sentence plus the numbers of the papers that support it.
CLAIMS = {
    "type": "array",
    "maxItems": 10,
    "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1, "maxLength": 600},
            "sources": {"type": "array", "maxItems": 10, "items": {"type": "integer"},
                        "description": "Paper ids from search results. Empty list = AI synthesis."},
        },
        "required": ["text", "sources"],
        "additionalProperties": False,
    },
}

REPORT_SECTIONS = ["main_methods", "datasets", "models", "metrics", "key_findings",
                   "limitations", "research_gaps", "future_directions"]

TOOLS = {
    "search_papers": {
        "description": "Search academic papers (Semantic Scholar, falling back to OpenAlex). "
                       "Returns real papers with ids you can cite as [id].",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2, "maxLength": 300, "description": "Search keywords."},
                "year_from": {"type": "integer", "minimum": 1900, "maximum": 2100,
                              "description": "Only papers published in this year or later."},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "read_pdf": {
        "description": "Get the passages of the user's uploaded PDF most relevant to a question.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "minLength": 2, "maxLength": 500}},
            "required": ["question"],
            "additionalProperties": False,
        },
    },
    "calculate": {
        "description": "Safely evaluate arithmetic. Supports + - * / ** %, parentheses and "
                       "sqrt, abs, round, mean, rmse(list, list), pct_change(old, new) (returns a percent).",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "minLength": 1, "maxLength": 200}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    "generate_report": {
        "description": "Write the final structured research report from the papers you retrieved. "
                       "A HUMAN MUST APPROVE this before it runs. Cite papers by id in 'sources'; "
                       "use an empty list for your own synthesis.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 1, "maxLength": 800,
                            "description": "Short note to the human explaining what you found and why a report is ready."},
                **{name: CLAIMS for name in REPORT_SECTIONS},
            },
            "required": ["summary", *REPORT_SECTIONS],
            "additionalProperties": False,
        },
    },
}

# Which tools each demo mode may use.
MODE_TOOLS = {
    "basic": [],
    "tool": ["search_papers", "read_pdf", "calculate"],
    "agent": ["search_papers", "read_pdf", "calculate", "generate_report"],
}


def tool_schemas(names):
    """The tool list in Responses-API format (flat: type/name/description/parameters)."""
    return [{"type": "function", "name": n, "description": TOOLS[n]["description"],
             "parameters": TOOLS[n]["parameters"]} for n in names]


# ---------------------------------------------------------------------------
# Validation (a tiny subset of JSON Schema - enough for our tools)
# ---------------------------------------------------------------------------

def _check(value, schema, path):
    kind = schema.get("type")
    type_ok = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
        "boolean": isinstance(value, bool),
    }.get(kind, True)
    if not type_ok:
        return [f"{path} must be of type {kind}"]

    errors = []
    if kind == "string":
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path} is too short")
        if len(value) > schema.get("maxLength", 10**9):
            errors.append(f"{path} is too long (max {schema['maxLength']} characters)")
    if kind in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} must be <= {schema['maximum']}")
    if kind == "array":
        if len(value) > schema.get("maxItems", 10**9):
            errors.append(f"{path} has too many items (max {schema['maxItems']})")
        for i, item in enumerate(value):
            errors += _check(item, schema.get("items", {}), f"{path}[{i}]")
    if kind == "object":
        props = schema.get("properties", {})
        errors += [f"{path}.{k} is required" for k in schema.get("required", []) if k not in value]
        if schema.get("additionalProperties") is False:
            errors += [f"{path}.{k} is not an allowed argument" for k in value if k not in props]
        for k, v in value.items():
            if k in props:
                errors += _check(v, props[k], f"{path}.{k}")
    return errors


def validate_call(name, raw_arguments, allowed):
    """Return (arguments_dict, None) if valid, else (None, error_message)."""
    if name not in TOOLS:
        return None, f"Unknown tool '{name}'. Available tools: {', '.join(allowed) or 'none'}."
    if name not in allowed:
        return None, f"Tool '{name}' is not available in this mode."
    try:
        args = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        return None, "Arguments are not valid JSON."
    if not isinstance(args, dict):
        return None, "Arguments must be a JSON object."
    errors = _check(args, TOOLS[name]["parameters"], "arguments")
    if errors:
        return None, "Invalid arguments: " + "; ".join(errors[:5])
    return args, None


# ---------------------------------------------------------------------------
# Execution. Each function returns (observation_for_model, log_lines_for_human).
# `memory` is the agent's session memory (papers found, uploaded PDF, ...).
# ---------------------------------------------------------------------------

def run_search_papers(args, memory, options):
    limit = min(args.get("max_results", options["max_papers"]), options["max_papers"])
    year_from = max([y for y in (args.get("year_from"), options.get("year_from")) if y], default=None)
    log = [f"Searching Semantic Scholar: '{args['query']}'" + (f" ({year_from}+)" if year_from else "")]

    result = paper_search.search_papers(args["query"], year_from, limit, api_key=options.get("s2_key", ""),
                                        openalex_key=options.get("openalex_key", ""))
    if result["fallback_reason"]:
        log.append(f"Semantic Scholar unavailable ({result['fallback_reason']}); used OpenAlex instead")

    new_papers = []
    known = {(p["doi"] or p["title"].lower()) for p in memory["papers"]}
    for paper in result["papers"]:
        key = paper["doi"] or paper["title"].lower()
        if key in known:
            paper = next(p for p in memory["papers"] if (p["doi"] or p["title"].lower()) == key)
        else:
            paper = {**paper, "id": len(memory["papers"]) + 1}
            memory["papers"].append(paper)
            known.add(key)
        new_papers.append(paper)

    log.append(f"Retrieved {len(new_papers)} papers from {result['source']}" + (" (cached)" if result.get("cached") else ""))
    observation = {
        "source": result["source"],
        "papers": [{
            "id": p["id"], "title": p["title"], "authors": p["authors"][:4], "year": p["year"],
            "venue": p["venue"], "abstract": p["abstract"][:1200], "doi": p["doi"], "url": p["url"],
        } for p in new_papers],
    }
    if not new_papers:
        observation["message"] = "No papers found. Try broader keywords or an earlier year_from."
    return observation, log


def run_read_pdf(args, memory, options):
    pdf = memory.get("pdf")
    if not pdf:
        return {"error": "No PDF has been uploaded."}, ["read_pdf called, but no PDF is uploaded"]
    picked = pdf_reader.top_chunks(pdf["chunks"], args["question"])
    observation = {
        "file": pdf["name"],
        "excerpts": [{"chunk": i + 1, "of": len(pdf["chunks"]), "text": text} for i, text in picked],
    }
    return observation, [f"Reading PDF '{pdf['name']}': selected {len(picked)} of {len(pdf['chunks'])} chunks"]


def run_calculate(args, memory, options):
    result = calculator.calculate(args["expression"])
    result = round(result, 10) if isinstance(result, float) else result
    return {"expression": args["expression"], "result": result}, [f"Calculated {args['expression']} = {result}"]


EXECUTORS = {
    "search_papers": run_search_papers,
    "read_pdf": run_read_pdf,
    "calculate": run_calculate,
    # generate_report is NOT here: agent.py runs it only after human approval.
}


def execute(name, args, memory, options):
    """Run a validated tool. Any tool failure becomes an error observation."""
    try:
        return EXECUTORS[name](args, memory, options)
    except (calculator.CalculatorError, paper_search.PaperSearchError, pdf_reader.PDFError) as e:
        return {"error": str(e)}, [f"{name} failed: {e}"]
    except Exception as e:  # last line of defence: never crash the app because of a tool
        return {"error": f"{name} failed unexpectedly ({type(e).__name__})."}, [f"{name} failed unexpectedly ({type(e).__name__})"]
