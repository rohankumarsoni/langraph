import time
import uuid
from datetime import datetime

import streamlit as st

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 1.5rem; max-width: 900px;}

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #4338ca 55%, #6d28d9 100%);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        color: #ffffff;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(67, 56, 202, 0.25);
    }
    .hero h1 {margin: 0; font-size: 1.7rem; font-weight: 700; color: #ffffff;}
    .hero p {margin: 0.35rem 0 0 0; opacity: 0.9; font-size: 0.97rem;}
    .badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.35);
        padding: 0.18rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-top: 0.8rem;
        font-weight: 600;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(128, 128, 128, 0.18);
        background: rgba(128, 128, 128, 0.06);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {border-right: 1px solid rgba(128, 128, 128, 0.2);}
    .side-title {font-size: 1.15rem; font-weight: 700; margin-bottom: 0.1rem;}
    .side-sub {font-size: 0.82rem; opacity: 0.7; margin-bottom: 0.8rem;}

    /* Stat cards */
    .stat {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 0.55rem 0.8rem;
        text-align: center;
    }
    .stat b {font-size: 1.2rem; display: block;}
    .stat span {font-size: 0.75rem; opacity: 0.7;}

    /* Suggestion buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #6366f1;
        color: #6366f1;
        transform: translateY(-1px);
    }
    .welcome-title {font-weight: 600; margin: 0.4rem 0 0.6rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# Load the LangGraph app once
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading college documents...")
def load_app():
    from backend import app
    return app


app = load_app()

PROGRAMMES = ["BCA", "BBA", "B.Com (H)"]

SUGGESTIONS = [
    ("💰", "What is my course fee?"),
    ("📅", "What is the attendance requirement?"),
    ("📝", "How does the grading system work?"),
    ("🔁", "What is the refund policy?"),
]


# ---------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------
def reset_chat():
    st.session_state.chat = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.started_at = datetime.now()


def stream_text(text: str):
    """Typing effect for the assistant's reply."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.015)


def build_transcript() -> str:
    lines = [
        f"College Assistant transcript - {st.session_state.programme}",
        f"Date: {datetime.now():%d %b %Y, %H:%M}",
        "-" * 50,
    ]
    for m in st.session_state.chat:
        who = "You" if m["role"] == "user" else "Assistant"
        lines.append(f"\n{who}:\n{m['content']}")
    return "\n".join(lines)


if "chat" not in st.session_state:
    reset_chat()

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-title">🎓 College Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-sub">Academics &amp; fee help, powered by AI</div>', unsafe_allow_html=True)

    st.selectbox(
        "Your programme",
        PROGRAMMES,
        key="programme",
        on_change=reset_chat,
        help="Changing the programme starts a fresh conversation.",
    )

    st.divider()

    user_msgs = sum(1 for m in st.session_state.chat if m["role"] == "user")
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="stat"><b>{user_msgs}</b><span>Questions</span></div>', unsafe_allow_html=True)
    c2.markdown(
        f'<div class="stat"><b>{st.session_state.started_at:%H:%M}</b><span>Started</span></div>',
        unsafe_allow_html=True,
    )

    st.write("")
    if st.button("🗑️ New chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.download_button(
        "⬇️ Download chat",
        data=build_transcript(),
        file_name=f"college_chat_{datetime.now():%Y%m%d_%H%M}.txt",
        mime="text/plain",
        use_container_width=True,
        disabled=not st.session_state.chat,
    )

    st.divider()
    st.caption(
        "Answers come from the official academics handbook and fee structure. "
        "For anything important, please confirm with the college office."
    )

programme = st.session_state.programme

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>College Assistant</h1>
        <p>Ask about academic rules, fees, or anything else about your course.</p>
        <span class="badge">📘 {programme} Student</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Decide the incoming query (typed or suggestion click)
# ---------------------------------------------------------------
typed_query = st.chat_input(f"Ask a question as a {programme} student...")
pending_query = st.session_state.pop("pending", None)
user_query = typed_query or pending_query

# ---------------------------------------------------------------
# Chat history / welcome state
# ---------------------------------------------------------------
for msg in st.session_state.chat:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🎓"):
        st.markdown(msg["content"])

if not st.session_state.chat and not user_query:
    st.markdown('<div class="welcome-title">👋 Try asking one of these:</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (icon, question) in enumerate(SUGGESTIONS):
        if cols[i % 2].button(f"{icon}  {question}", key=f"sug_{i}", use_container_width=True):
            st.session_state.pending = question
            st.rerun()

# ---------------------------------------------------------------
# Handle the new query
# ---------------------------------------------------------------
if user_query:
    st.session_state.chat.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🎓"):
        try:
            with st.spinner("Searching college documents..."):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = app.invoke(
                    {
                        "programme": programme,
                        "messages": [("human", user_query)],
                    },
                    config=config,
                )
                answer = result["messages"][-1].content
            st.write_stream(stream_text(answer))
        except Exception as e:
            answer = f"⚠️ Something went wrong while fetching your answer.\n\n`{e}`"
            st.error(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer})
    st.rerun()