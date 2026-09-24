"""Mini-Muse Streamlit UI - a chat-style interface.  Run with:  streamlit run app.py

Streamlit re-runs this whole script on every click. That's why:
  - the agent state lives in st.session_state["agent"] (a plain dict), and
  - Muse Spark is only called when you send a message, never while just drawing the page.
"""
import copy
import html
from datetime import date
from pathlib import Path

import streamlit as st

import agent
import config
import meta_client
import pdf_reader

MODES = {
    "Agent": ("agent", "Decides, uses tools, observes, repeats, asks for approval"),
    "Tool-using LLM": ("tool", "One tool call, then an answer"),
    "Basic LLM": ("basic", "Answers from memory, no tools"),
}
ASSISTANT_AVATAR = ":material/auto_awesome:"
SUGGESTIONS = [
    ("Literature review", "Find recent papers about physics-informed solar power forecasting and identify the main research gaps."),
    ("Quick calculation", "Calculate the percentage increase from 100 to 125."),
    ("Explore a topic", "Find papers about graph neural networks for traffic prediction."),
]
LOG_ICONS = [("Searching", "🔎"), ("Retrieved", "📚"), ("Reading PDF", "📄"), ("Calculated", "🧮"),
             ("chose tool", "🛠️"), ("asking Muse Spark", "💭"), ("Waiting", "⏸️"), ("approved", "✅"),
             ("rejected", "✋"), ("Report saved", "💾"), ("Writing", "✍️"), ("failed", "⚠️"),
             ("refused", "⛔"), ("unavailable", "↪️"), ("Stopped", "⏹️"), ("Final answer", "✨")]

st.set_page_config(page_title="Mini-Muse", page_icon="✦", layout="centered", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Styling (plain CSS; only changes how things look)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
/* Inter everywhere - but never on Streamlit's icon font, or icons turn into words like "check" */
.stApp :is(p, li, h1, h2, h3, h4, label, button, input, textarea, div, span, a):not([data-testid^="stIcon"]):not([data-testid^="stExpanderIcon"]) {
    font-family: 'Inter', system-ui, sans-serif; }
#MainMenu, footer, [data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 820px; padding-top: 2.2rem; padding-bottom: 7rem; }

/* sidebar */
[data-testid="stSidebar"] { background: #F7F7F8; border-right: 1px solid #ECECF1; }
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
.brand { display:flex; align-items:center; gap:.55rem; font-weight:700; font-size:1.25rem; color:#1F2328; margin-bottom:.1rem; }
.brand-mark { width:30px; height:30px; border-radius:9px; display:grid; place-items:center; color:#fff; font-size:1rem;
              background: linear-gradient(135deg,#4F46E5,#7C3AED); }
.brand-sub { color:#6B7280; font-size:.8rem; margin: 0 0 1rem 2.4rem; }
.side-label { font-size:.72rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:#8B8F98; margin:1.1rem 0 .3rem; }
.status-pill { display:flex; align-items:center; gap:.5rem; padding:.55rem .75rem; border-radius:10px; font-size:.82rem;
               background:#fff; border:1px solid #ECECF1; color:#374151; }
.dot { width:8px; height:8px; border-radius:50%; flex:none; }
.dot.live { background:#10B981; box-shadow:0 0 0 3px #D1FAE5; }
.dot.mock { background:#F59E0B; box-shadow:0 0 0 3px #FEF3C7; }
.meta-line { font-size:.75rem; color:#8B8F98; margin-top:.45rem; line-height:1.5; }
.meta-line code { font-size:.72rem; background:#EEF0F3; color:#4B5563; padding:1px 5px; border-radius:5px; }

/* hero (empty chat) */
.hero { text-align:center; margin: 9vh 0 2rem; }
.hero-mark { width:52px; height:52px; border-radius:15px; margin:0 auto 1rem; display:grid; place-items:center;
             color:#fff; font-size:1.6rem; background: linear-gradient(135deg,#4F46E5,#7C3AED); }
.hero h1 { font-size:2rem; font-weight:650; color:#1F2328; margin:0; padding:0; }
.hero p { color:#6B7280; margin:.5rem 0 0; }
.st-key-suggestions .stButton button { height:100%; min-height:92px; text-align:left; white-space:normal;
    border:1px solid #E5E7EB; border-radius:14px; background:#fff; color:#374151; padding:.8rem 1rem; font-size:.86rem;
    line-height:1.4; transition: all .15s ease; }
.st-key-suggestions .stButton button:hover { border-color:#C7D2FE; background:#F8F8FF; color:#1F2328; }
.st-key-suggestions .stButton button p { white-space:normal; overflow:visible; text-overflow:clip; text-align:left; }

/* mode banner + mock banner */
.topbar { display:flex; justify-content:space-between; align-items:center; gap:.75rem; margin-bottom:1rem; flex-wrap:wrap; }
.chip { display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .7rem; border-radius:999px; font-size:.78rem;
        font-weight:500; border:1px solid #E5E7EB; background:#fff; color:#374151; }
.chip.mock { background:#FFFBEB; border-color:#FDE68A; color:#92400E; }
.chip.live { background:#ECFDF5; border-color:#A7F3D0; color:#065F46; }

/* chat messages */
[data-testid="stChatMessage"] { background: transparent; padding: .35rem 0; gap: .8rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { justify-content: flex-end; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background:#F4F4F5; border-radius:18px; padding:.65rem 1rem; max-width:80%; margin-left:auto; margin-right:0 !important; flex: 0 1 auto; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] { display:none; }
.attach { display:inline-flex; gap:.35rem; align-items:center; font-size:.78rem; color:#4B5563; background:#fff;
          border:1px solid #E5E7EB; border-radius:8px; padding:.15rem .5rem; margin-top:.4rem; }

/* headings inside answers/reports: document-sized, not page-sized */
[data-testid="stChatMessageContent"] h1 { font-size:1.35rem; font-weight:650; margin-top:.4rem; }
[data-testid="stChatMessageContent"] h2 { font-size:1.08rem; font-weight:620; margin:1.1rem 0 .3rem; padding:0; }
[data-testid="stChatMessageContent"] h3 { font-size:.98rem; font-weight:600; }
[data-testid="stChatMessageContent"] blockquote { font-size:.85rem; color:#6B7280; border-left:3px solid #E5E7EB; }
[data-testid="stChatMessageContent"] li, [data-testid="stChatMessageContent"] p { font-size:.95rem; line-height:1.6; }

/* sources */
.src { display:flex; gap:.7rem; padding:.7rem .85rem; border:1px solid #ECECF1; border-radius:12px; margin-bottom:.5rem; background:#fff; }
.src-n { flex:none; width:24px; height:24px; border-radius:7px; background:#EEF2FF; color:#4338CA; font-size:.75rem;
         font-weight:600; display:grid; place-items:center; }
.src-t { font-weight:550; font-size:.9rem; color:#1F2328; text-decoration:none; line-height:1.35; }
.src-t:hover { color:#4F46E5; text-decoration:underline; }
.src-m { font-size:.78rem; color:#6B7280; margin-top:.2rem; }
.src-a { font-size:.8rem; color:#4B5563; margin-top:.35rem; line-height:1.45; }

/* approval card */
.approve-head { display:flex; align-items:center; gap:.5rem; font-weight:600; color:#1F2328; }
.approve-sub { color:#6B7280; font-size:.86rem; margin:.2rem 0 .6rem; }
.usage { font-size:.74rem; color:#9CA3AF; margin-top:.3rem; }

/* chat input */
[data-testid="stChatInput"] { border-radius:16px; }
[data-testid="stBottomBlockContainer"] { max-width: 820px; padding-bottom: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
settings = config.load_settings()
st.session_state.setdefault("client", meta_client.make_client(settings))
st.session_state.setdefault("agent", agent.new_state())
st.session_state.setdefault("turns", [])        # finished earlier runs, shown above the current one
st.session_state.setdefault("pdf", None)
st.session_state.setdefault("pending_prompt", None)
client = st.session_state.client
state = st.session_state.agent


def new_chat():
    st.session_state.agent = agent.new_state()
    st.session_state.turns = []
    st.session_state.pdf = None


def esc(text) -> str:
    """Escape text from APIs before putting it inside our HTML."""
    return html.escape(str(text or ""))


def icon_for(line: str) -> str:
    return next((icon for key, icon in LOG_ICONS if key.lower() in line.lower()), "•")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">✦</div>Mini-Muse</div>'
                '<div class="brand-sub">Educational AI research agent</div>', unsafe_allow_html=True)
    if st.button("＋  New chat", use_container_width=True, type="primary"):
        new_chat()
        st.rerun()

    st.markdown('<div class="side-label">Mode</div>', unsafe_allow_html=True)
    mode_label = st.radio("Mode", list(MODES), captions=[m[1] for m in MODES.values()], label_visibility="collapsed")
    mode = MODES[mode_label][0]

    st.markdown('<div class="side-label">Search</div>', unsafe_allow_html=True)
    max_papers = st.slider("Max papers per search", 1, 10, config.DEFAULT_MAX_PAPERS)
    year_choice = st.selectbox("Published from", ["Any year"] + list(range(date.today().year, 2009, -1)))
    web_search = st.toggle("Web search", value=False, disabled=client.mode != "live",
                           help="Meta's built-in web_search tool (Responses API). Needs a live API key.")

    if st.session_state.pdf:
        st.markdown('<div class="side-label">Attached PDF</div>', unsafe_allow_html=True)
        pdf = st.session_state.pdf
        st.caption(f"📄 {pdf['name']}  \n{pdf['pages']} pages · {pdf['words']:,} words")
        if st.button("Remove PDF", use_container_width=True):
            st.session_state.pdf = None
            st.rerun()

    st.markdown('<div class="side-label">Connection</div>', unsafe_allow_html=True)
    if client.mode == "live":
        st.markdown('<div class="status-pill"><span class="dot live"></span>Live · Meta Model API</div>', unsafe_allow_html=True)
    else:
        reason = "live access refused" if client.fallback_error else client.mock_reason
        st.markdown(f'<div class="status-pill"><span class="dot mock"></span>Mock mode · {esc(reason)}</div>',
                    unsafe_allow_html=True)
        if client.fallback_error:
            st.caption(client.fallback_error)
    st.markdown(f'<div class="meta-line">Model <code>{esc(settings.model)}</code><br>'
                f'Reasoning effort <code>{esc(settings.reasoning_effort)}</code></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Rendering one question + answer
# ---------------------------------------------------------------------------
def render_sources(papers, citations):
    with st.expander(f"📚  Sources · {len(papers) + len(citations)}", expanded=False):
        for p in papers:
            authors = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
            meta = " · ".join(x for x in [authors, str(p["year"] or ""), p["venue"], f"DOI {p['doi']}" if p["doi"] else ""] if x)
            abstract = p["abstract"] if len(p["abstract"]) < 320 else p["abstract"][:320].rsplit(" ", 1)[0] + "…"
            title = (f'<a class="src-t" href="{esc(p["url"])}" target="_blank">{esc(p["title"])}</a>'
                     if p["url"] else f'<span class="src-t">{esc(p["title"])}</span>')
            st.markdown(f'<div class="src"><div class="src-n">{p["id"]}</div><div>{title}'
                        f'<div class="src-m">{esc(meta)}</div><div class="src-a">{esc(abstract)}</div></div></div>',
                        unsafe_allow_html=True)
        for c in citations:
            st.markdown(f'<div class="src"><div class="src-n">🌐</div><div><a class="src-t" href="{esc(c["url"])}" '
                        f'target="_blank">{esc(c["title"])}</a></div></div>', unsafe_allow_html=True)


def render_approval(run):
    draft = run["pending_report"]["arguments"]
    with st.container(border=True):
        st.markdown('<div class="approve-head">⏸️ Approval required</div>'
                    '<div class="approve-sub">Muse Spark wants to write the final report. Nothing is written until you decide.</div>',
                    unsafe_allow_html=True)
        st.markdown(draft["summary"])
        with st.expander("Preview key findings and research gaps"):
            for section in ("key_findings", "research_gaps"):
                st.markdown(f"**{section.replace('_', ' ').capitalize()}**")
                for claim in draft[section]:
                    cites = "".join(f"[{i}]" for i in claim["sources"]) or "*(AI synthesis)*"
                    st.markdown(f"- {claim['text']} {cites}")
        col1, col2, _ = st.columns([1, 1, 2])
        if col1.button("Approve", type="primary", use_container_width=True, key="approve"):
            agent.approve(run)
            st.rerun()
        if col2.button("Reject", use_container_width=True, key="reject"):
            agent.reject(run)
            st.rerun()


def render_turn(run, index, is_current):
    with st.chat_message("user"):
        st.markdown(run["goal"])
        if run["pdf"] and run["mode"] != "basic":
            st.markdown(f'<span class="attach">📄 {esc(run["pdf"]["name"])}</span>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        if run["log"]:
            failed = run["status"] == agent.ERROR
            label = (f"Stopped with an error" if failed else
                     "Waiting for your approval" if run["status"] == agent.WAITING_APPROVAL else
                     f"Worked through {len(run['log'])} steps")
            with st.status(label, state="error" if failed else "complete", expanded=False):
                for line in run["log"]:
                    st.markdown(f"{icon_for(line)}&nbsp; {esc(line)}", unsafe_allow_html=True)

        if run["papers"] or run["citations"]:
            render_sources(run["papers"], run["citations"])

        if run["status"] == agent.ERROR:
            st.error(run["error"])
        elif run["status"] == agent.WAITING_APPROVAL and is_current:
            render_approval(run)
        elif run["report_markdown"]:              # the report is shown as a document card
            with st.container(border=True):
                st.markdown(run["report_markdown"])
            st.download_button("Download report (.md)", run["report_markdown"], key=f"dl_{index}",
                               icon=":material/download:", file_name=Path(run["report_path"]).name, mime="text/markdown")
            st.caption(f"Saved to outputs/{Path(run['report_path']).name}")
        elif run["final_answer"]:
            st.markdown(run["final_answer"])
        u = run["usage"]
        if client.mode == "live" and u["input_tokens"]:
            st.markdown(f'<div class="usage">{u["input_tokens"]:,} input · {u["output_tokens"]:,} output tokens '
                        f'({u["reasoning_tokens"]:,} reasoning)</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
has_chat = bool(st.session_state.turns or state["status"] != agent.IDLE)

badge = ('<span class="chip live">● Live · Muse Spark</span>' if client.mode == "live"
         else '<span class="chip mock">● Mock mode — scripted model, real tools</span>')
st.markdown(f'<div class="topbar"><span class="chip">{esc(mode_label)} mode</span>{badge}</div>', unsafe_allow_html=True)

if not has_chat:
    st.markdown('<div class="hero"><div class="hero-mark">✦</div><h1>What would you like to research?</h1>'
                '<p>Ask a research question, attach a PDF, or try an example.</p></div>', unsafe_allow_html=True)
    cols = st.container(key="suggestions").columns(len(SUGGESTIONS))
    for col, (title, prompt) in zip(cols, SUGGESTIONS):
        if col.button(f"**{title}** · {prompt}", key=f"suggest_{title}", use_container_width=True):
            st.session_state.pending_prompt = prompt
            st.rerun()

for i, old in enumerate(st.session_state.turns):
    render_turn(old, i, is_current=False)
if state["status"] != agent.IDLE:
    render_turn(state, len(st.session_state.turns), is_current=True)

# ---------------------------------------------------------------------------
# Chat input (text + optional PDF attachment)
# ---------------------------------------------------------------------------
waiting = state["status"] == agent.WAITING_APPROVAL
submitted = st.chat_input(
    "Approve or reject the report above to continue" if waiting else "Ask a research question…",
    accept_file=True, file_type=["pdf"], disabled=waiting,
)

prompt = None
if submitted:
    for upload in submitted.files:
        try:
            path = pdf_reader.save_upload(upload.name, upload.getvalue(), config.UPLOADS_DIR)
            st.session_state.pdf = pdf_reader.load_pdf(path)
            st.toast(f"Attached {st.session_state.pdf['name']}", icon="📄")
        except pdf_reader.PDFError as e:
            st.toast(str(e), icon="⚠️")
    prompt = (submitted.text or "").strip() or None
    if not prompt and submitted.files:
        st.rerun()                     # only a file was attached: show it in the sidebar
elif st.session_state.pending_prompt:
    prompt, st.session_state.pending_prompt = st.session_state.pending_prompt, None

if prompt:
    if state["status"] != agent.IDLE:                     # keep the finished run visible above
        st.session_state.turns.append(copy.deepcopy(state))
    run = agent.new_state(pdf=st.session_state.pdf)
    st.session_state.agent = run
    options = {
        "max_papers": max_papers,
        "year_from": None if year_choice == "Any year" else int(year_choice),
        "use_web_search": web_search,
        "model": settings.model if client.mode == "live" else "mock model",
        "s2_key": settings.semantic_scholar_key,
        "openalex_key": settings.openalex_key,
    }
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.status("Working…", expanded=True) as progress:
            agent.start(run, prompt, mode, options, client,
                        on_log=lambda line: st.markdown(f"{icon_for(line)}&nbsp; {esc(line)}", unsafe_allow_html=True))
            progress.update(label="Done", state="complete", expanded=False)
    st.rerun()
