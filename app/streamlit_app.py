import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH so "src" imports work when running from /app
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import io
import streamlit as st
from reportlab.pdfgen import canvas

from src.config import MODEL_MODE
from src.chat_engine import ask_hr_bot

st.set_page_config(page_title="HR LLM Chatbot", layout="wide")

st.title("HR Dataset Chatbot (Text-to-SQL + Memory)")
st.caption("Ask questions in plain English. The bot generates SQL, queries SQLite, then explains results.")

# Session state: memory
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {role, content}
if "debug" not in st.session_state:
    st.session_state.debug = True

# Sidebar
with st.sidebar:
    st.subheader("Settings")

    # ✅ Runtime model switch (overrides .env MODEL_MODE)
    mode = st.selectbox(
        "Model mode",
        ["groq", "local"],
        index=0 if MODEL_MODE == "groq" else 1,
    )

    st.write(f"Current mode from .env: **{MODEL_MODE}**")
    st.session_state.debug = st.toggle("Show SQL + table preview (debug)", value=True)

    st.divider()
    st.subheader("Export chat")

    # Export TXT
    if st.button("Download TXT"):
        txt = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button("Save transcript.txt", txt, file_name="transcript.txt")

    # Export PDF
    if st.button("Download PDF"):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer)
        y = 800
        c.setFont("Helvetica", 10)
        for m in st.session_state.messages:
            lines = (f"{m['role'].upper()}: {m['content']}").split("\n")
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
        st.download_button("Save transcript.pdf", buffer, file_name="transcript.pdf")

# Display conversation
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask something about employees, attrition, income, departments…")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sql, df = ask_hr_bot(
                    prompt,
                    st.session_state.messages,
                    mode  # ✅ pass selected mode at runtime
                )

                st.markdown(answer)

                if st.session_state.debug:
                    st.markdown("**SQL generated:**")
                    st.code(sql, language="sql")
                    st.markdown("**Result preview:**")
                    st.dataframe(df.head(50))

                # Store assistant message (only the answer text for memory)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                err = f"Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
