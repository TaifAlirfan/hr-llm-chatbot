import io
import sys
from pathlib import Path

import streamlit as st
from reportlab.pdfgen import canvas

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import MODEL_MODE
from src.chat_engine import ask_hr_bot

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="HR LLM Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Dark Blue modern CSS
# -----------------------------
st.markdown(
    """
<style>
/* ---------- Global dark theme overrides ---------- */
:root {
  --bg: #0B1220;
  --panel: #0F1B2D;
  --panel2: #0C1728;
  --border: rgba(148,163,184,0.14);
  --text: #E5E7EB;
  --muted: rgba(229,231,235,0.68);
  --blue: #3B82F6;
  --blue2: #2563EB;
  --shadow: rgba(0,0,0,0.35);
}

/* App background */
.stApp {
  background: radial-gradient(1000px 700px at 18% 8%, rgba(37,99,235,0.28) 0%, var(--bg) 55%, #070B14 100%) !important;
  color: var(--text);
}

/* Page container */
.block-container{
  padding-top: 1.25rem !important;
  padding-bottom: 2.5rem !important;
  max-width: 1200px !important;
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, var(--panel), var(--bg)) !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] *{
  color: var(--text) !important;
}

/* Remove Streamlit header whitespace */
header[data-testid="stHeader"]{
  background: transparent !important;
}

/* Labels + help text */
label, .stCaption, .stMarkdown p, .stMarkdown span {
  color: var(--text) !important;
}
small, .stCaption, .stMarkdown .small-note { color: var(--muted) !important; }

/* Buttons */
.stButton > button{
  width: 100%;
  border-radius: 14px !important;
  padding: 0.65rem 0.9rem !important;
  background: rgba(15,27,45,0.65) !important;
  color: var(--text) !important;
  border: 1px solid rgba(59,130,246,0.25) !important;
  box-shadow: 0 8px 22px var(--shadow);
}
.stButton > button:hover{
  border-color: rgba(59,130,246,0.65) !important;
  transform: translateY(-1px);
}

/* Inputs */
[data-testid="stChatInput"]{
  background: transparent !important;
}
[data-testid="stChatInput"] textarea{
  border-radius: 16px !important;
  background: rgba(15,27,45,0.65) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}
[data-testid="stChatInput"] textarea::placeholder{
  color: rgba(229,231,235,0.45) !important;
}

/* Toggle / radio cards */
div[role="radiogroup"] label,
div[data-testid="stToggle"] label{
  color: var(--text) !important;
}
div[data-testid="stToggle"]{
  background: transparent !important;
}

/* Chat bubbles */
div[data-testid="stChatMessage"]{
  border-radius: 16px;
  border: 1px solid var(--border);
  background: rgba(15,27,45,0.45);
  box-shadow: 0 10px 26px var(--shadow);
}
div[data-testid="stChatMessage"] *{
  color: var(--text) !important;
}

/* Dataframe / code */
div[data-testid="stDataFrame"]{
  background: rgba(15,27,45,0.45) !important;
  border: 1px solid var(--border) !important;
}
code, pre{
  background: rgba(2,6,23,0.55) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 12px !important;
}

/* ---------- Header card ---------- */
.hero {
  border-radius: 20px;
  padding: 18px 18px 16px 18px;
  background: linear-gradient(135deg, rgba(37,99,235,0.22), rgba(15,27,45,0.55));
  border: 1px solid rgba(59,130,246,0.22);
  box-shadow: 0 14px 34px var(--shadow);
  overflow: hidden;   /* prevents cropping artifacts */
}
.hero-row{
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.hero-title{
  font-size: 32px;
  font-weight: 850;
  line-height: 1.1;
  letter-spacing: -0.4px;
  margin: 0;
  padding: 0;
  color: var(--text);
}
.hero-sub{
  margin-top: 8px;
  font-size: 13px;
  color: var(--muted);
}

/* Pills (no emojis) */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  background: rgba(2,6,23,0.35);
  border: 1px solid rgba(59,130,246,0.25);
  color: var(--text);
}

/* KPI */
.kpi-wrap {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  margin-top: 14px;
}
.kpi {
  border-radius: 16px;
  padding: 12px;
  background: rgba(15,27,45,0.50);
  border: 1px solid var(--border);
}
.kpi-label { font-size: 12px; color: var(--muted); }
.kpi-value { font-size: 20px; font-weight: 850; margin-top: 2px; color: var(--text); }

@media (max-width: 1100px) {.kpi-wrap {grid-template-columns: repeat(2, 1fr);} }
@media (max-width: 600px) {.kpi-wrap {grid-template-columns: 1fr;} }

hr {opacity: 0.18;}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug" not in st.session_state:
    st.session_state.debug = False
if "ui_mode" not in st.session_state:
    st.session_state.ui_mode = (MODEL_MODE or "groq").strip().lower()
if "show_memory" not in st.session_state:
    st.session_state.show_memory = False
if "memory_turns" not in st.session_state:
    st.session_state.memory_turns = 8

# ✅ NEW: used to "submit" smart prompts automatically
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


def format_memory(messages, max_turns: int = 8) -> str:
    recent = messages[-max_turns:]
    if not recent:
        return "(empty)"
    lines = []
    for m in recent:
        role = m.get("role", "").upper()
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def build_pdf_bytes(messages) -> io.BytesIO:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    y = 800
    c.setFont("Helvetica", 10)

    for m in messages:
        header = f"{m['role'].upper()}: "
        text = m["content"] or ""
        lines = (header + text).split("\n")

        for line in lines:
            c.drawString(40, y, line[:120])
            y -= 14
            if y < 60:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = 800
        y -= 10

    c.save()
    buffer.seek(0)
    return buffer


# -----------------------------
# Header (dark + blue)
# -----------------------------
mode_label = "Groq API" if st.session_state.ui_mode == "groq" else "Local Model"
memory_label = "Ready" if len(st.session_state.messages) == 0 else "Active"

total_turns = len(st.session_state.messages)
user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")

st.markdown(
    f"""
<div class="hero">
  <div class="hero-row">
    <div>
      <h1 class="hero-title">HR Dataset Chatbot</h1>
      <div class="hero-sub">Ask questions in plain English — get verified answers from SQLite.</div>
    </div>
    <div>
      <span class="pill">Memory: <b>{memory_label}</b></span>
      <span class="pill">Mode: <b>{mode_label}</b></span>
    </div>
  </div>

  <div class="kpi-wrap">
    <div class="kpi">
      <div class="kpi-label">Session turns</div>
      <div class="kpi-value">{total_turns}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">User questions</div>
      <div class="kpi-value">{user_turns}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Debug</div>
      <div class="kpi-value">{"On" if st.session_state.debug else "Off"}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Status</div>
      <div class="kpi-value">Online</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")


# -----------------------------
# Sidebar (controls + memory + smart prompts)
# -----------------------------
with st.sidebar:
    st.subheader("Controls")

    st.session_state.ui_mode = st.radio(
        "Model mode",
        options=["groq", "local"],
        index=0 if st.session_state.ui_mode == "groq" else 1,
        help="Groq is faster and more accurate. Local is offline but weaker.",
    )

    st.session_state.debug = st.toggle("Show SQL + table preview (debug)", value=st.session_state.debug)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()
    with col2:
        st.session_state.show_memory = st.toggle("Show memory", value=st.session_state.show_memory)

    if st.session_state.show_memory:
        st.session_state.memory_turns = st.slider("Memory turns", 2, 20, st.session_state.memory_turns)
        st.caption("This shows the last turns kept in session memory.")
        st.code(format_memory(st.session_state.messages, st.session_state.memory_turns), language="text")

    st.divider()

    st.subheader("Smart prompts")
    st.caption("Click a prompt to instantly run it.")
    quick = [
        "How many employees are there?",
        "How many employees are above 39?",
        "Show employee count by department.",
        "Which department has the highest average JobSatisfaction?",
        "What is the attrition rate by department?",
        "Compare average MonthlyIncome across departments.",
    ]

    # ✅ IMPORTANT CHANGE:
    # Instead of appending to messages (which doesn't trigger the assistant),
    # we set pending_prompt and rerun so it goes through the normal pipeline.
    for q in quick:
        if st.button(q, use_container_width=True):
            st.session_state.pending_prompt = q
            st.rerun()

    st.divider()

    st.subheader("Export chat")
    txt = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages]) or ""
    st.download_button("Download TXT", txt, file_name="transcript.txt", use_container_width=True)

    pdf_buffer = build_pdf_bytes(st.session_state.messages)
    st.download_button("Download PDF", pdf_buffer, file_name="transcript.pdf", use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small-note'>Tip: Use Groq for best answers; use Local to show offline capability.</div>",
        unsafe_allow_html=True,
    )

# -----------------------------
# Conversation rendering
# -----------------------------
if len(st.session_state.messages) == 0:
    st.info("Ask a question about employees, attrition, satisfaction, income, overtime, departments…")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -----------------------------
# Input + response
# -----------------------------
prompt = st.chat_input("Ask something about the HR dataset…")

# ✅ If user clicked a Smart Prompt, treat it as a submitted prompt
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sql, df = ask_hr_bot(
                    prompt,
                    st.session_state.messages,
                    st.session_state.ui_mode,
                )

                st.markdown(answer)

                if st.session_state.debug:
                    st.markdown("**SQL generated:**")
                    st.code(sql, language="sql")
                    st.markdown("**Result preview:**")
                    st.dataframe(df.head(50))

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error: {e}")
