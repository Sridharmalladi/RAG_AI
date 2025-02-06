import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

st.set_page_config(page_title="CodeMind AI", page_icon="💻")

st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #1E1E2E 0%, #2D2D44 100%);
        color: #E0E0E0;
    }
    .sidebar .sidebar-content {
        background: rgba(30, 30, 46, 0.6);
    }
    h1, h2, h3 {
        color: #7EB2FF !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background: rgba(30, 30, 46, 0.6) !important;
        border: 2px solid #7EB2FF !important;
        color: white !important;
    }
    div[role="listbox"] div {
        background: #1E1E2E !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("💻 CodeMind AI")
st.caption("Your Intelligent Programming Assistant")

with st.sidebar:
    st.header("⚙️ Settings")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )

llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

# Improved system prompt
system_prompt = SystemMessagePromptTemplate.from_template(
    """You are an expert programming assistant. Follow these rules:
    1. Provide clear, working code examples
    2. Explain complex concepts simply
    3. Include error handling in code
    4. Give step-by-step explanations
    Always write production-ready, clean code."""
)

if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm CodeMind. How can I help you with programming? 💻"}]

chat_container = st.container()

with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

user_query = st.chat_input("Type your programming question...")

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.rerun()