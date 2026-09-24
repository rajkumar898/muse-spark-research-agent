"""Builds the structured research report (Markdown).

Rules:
- The Sources list is written by our code from the papers the search tool
  actually returned. The model cannot add a source.
- Every claim cites retrieved papers by number, e.g. [1][3]. If a claim has no
  valid citation it is labelled "AI synthesis" so the reader knows to verify it.
"""
import re
from datetime import datetime
from pathlib import Path

SECTIONS = [
    ("main_methods", "Main methods"),
    ("datasets", "Datasets"),
    ("models", "Models"),
    ("metrics", "Metrics"),
    ("key_findings", "Key findings"),
    ("limitations", "Limitations"),
    ("research_gaps", "Research gaps"),
    ("future_directions", "Future directions"),
]


def format_claim(claim: dict, valid_ids: set[int]) -> str:
    text = str(claim.get("text", "")).strip()
    ids = sorted({i for i in claim.get("sources", []) if i in valid_ids})
    if ids:
        return f"- {text} " + "".join(f"[{i}]" for i in ids)
    return f"- {text} *(AI synthesis)*"


def format_source(paper: dict) -> str:
    authors = paper.get("authors") or []
    author_text = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") if authors else "Unknown authors"
    parts = [f"- [{paper['id']}] {author_text} ({paper.get('year') or 'n.d.'}). *{paper['title']}*."]
    if paper.get("venue"):
        parts.append(f"{paper['venue']}.")
    if paper.get("doi"):
        parts.append(f"DOI: {paper['doi']}.")
    if paper.get("url"):
        parts.append(paper["url"])
    return " ".join(parts)


def build_report(question: str, papers: list[dict], sections: dict, model: str, web_citations=None) -> str:
    valid_ids = {p["id"] for p in papers}
    lines = [
        "# Mini-Muse Research Report",
        "",
        f"*Generated {datetime.now():%Y-%m-%d %H:%M} by Mini-Muse ({model}). Human-approved before writing.*",
        "",
        "> Citation key: **[n]** = retrieved paper n in the Sources list. "
        "***AI synthesis*** = the model's own synthesis, not backed by a retrieved paper - verify before use.",
        "",
        "## Research question",
        "",
        question.strip(),
        "",
        "## Sources",
        "",
    ]
    lines += [format_source(p) for p in papers] or ["- No papers were retrieved."]
    for cite in web_citations or []:
        lines.append(f"- Web: [{cite['title']}]({cite['url']})")

    for key, heading in SECTIONS:
        lines += ["", f"## {heading}", ""]
        claims = [c for c in sections.get(key) or [] if str(c.get("text", "")).strip()]
        lines += [format_claim(c, valid_ids) for c in claims] or ["- Not covered by the retrieved sources."]
    return "\n".join(lines) + "\n"


def save_report(markdown: str, outputs_dir: Path, question: str) -> Path:
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:50] or "report"
    path = outputs_dir / f"report_{datetime.now():%Y%m%d_%H%M%S}_{slug}.md"
    path.write_text(markdown, encoding="utf-8")
    return path
