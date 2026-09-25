"""Run Mini-Muse in the terminal, without the Streamlit dashboard.

Same agent, same tools, same human approval - just printed as text.

    python cli.py "Calculate the percentage increase from 100 to 125."
    python cli.py "Find recent papers about physics-informed solar power forecasting and identify the main research gaps."
    python cli.py --mode basic "Calculate the percentage increase from 100 to 125."
    python cli.py --mode tool --pdf paper.pdf "What dataset and metrics does this paper use?"
"""
import argparse
import sys
from pathlib import Path

import agent
import config
import meta_client
import pdf_reader

MODES = {"agent": "Agent", "tool": "Tool-using LLM", "basic": "Basic LLM"}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles: never crash on symbols

    parser = argparse.ArgumentParser(description="Mini-Muse research agent (terminal version)")
    parser.add_argument("question", help="your question or research goal, in quotes")
    parser.add_argument("--mode", choices=list(MODES), default="agent", help="demo mode (default: agent)")
    parser.add_argument("--pdf", help="path to a PDF the agent may read")
    parser.add_argument("--max-papers", type=int, default=config.DEFAULT_MAX_PAPERS, choices=range(1, 11), metavar="1-10")
    args = parser.parse_args()

    settings = config.load_settings()
    client = meta_client.make_client(settings)
    print(f"\nMini-Muse · {MODES[args.mode]} mode · model {settings.model}")
    print("Connection: LIVE (Meta Model API)" if client.mode == "live" else f"Connection: MOCK MODE ({client.mock_reason})")

    state = agent.new_state()
    if args.pdf:
        try:
            state["pdf"] = pdf_reader.load_pdf(Path(args.pdf))
            print(f"PDF: {state['pdf']['name']} ({state['pdf']['pages']} pages, {len(state['pdf']['chunks'])} chunks)")
        except (pdf_reader.PDFError, OSError) as e:
            sys.exit(f"Cannot use that PDF: {e}")

    print(f"\nYou: {args.question}\n\nAgent steps:")
    options = {"max_papers": args.max_papers, "year_from": None, "use_web_search": False,
               "model": settings.model if client.mode == "live" else "mock model",
               "s2_key": settings.semantic_scholar_key, "openalex_key": settings.openalex_key}
    agent.start(state, args.question, args.mode, options, client, on_log=lambda line: print(f"  • {line}"))

    if state["papers"]:
        print(f"\nSources ({len(state['papers'])}):")
        for p in state["papers"]:
            print(f"  [{p['id']}] {p['title']} ({p['year'] or 'n.d.'}) {('DOI ' + p['doi']) if p['doi'] else p['url']}")

    for result in state["tool_results"]:                       # show what the PDF tool actually read
        for ex in (result["observation"] or {}).get("excerpts", []) if result["tool"] == "read_pdf" else []:
            snippet = ex["text"] if len(ex["text"]) <= 400 else ex["text"][:400] + "…"
            print(f"\nPDF excerpt (chunk {ex['chunk']} of {ex['of']}):\n  {snippet}")

    if state["status"] == agent.WAITING_APPROVAL:            # human approval, enforced by agent.py
        print(f"\nAPPROVAL REQUIRED - {state['pending_report']['arguments']['summary']}")
        answer = input("Approve writing the report? [y/N] ").strip().lower()
        agent.approve(state) if answer in ("y", "yes") else agent.reject(state)
        print(f"  • {state['log'][-1]}")

    if state["status"] == agent.ERROR:
        sys.exit(f"\nError: {state['error']}")
    print("\n" + "=" * 70 + "\n" + state["final_answer"] + "\n" + "=" * 70)
    if state["report_path"]:
        print(f"Report saved to: {state['report_path']}")


if __name__ == "__main__":
    main()
