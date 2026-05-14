from __future__ import annotations
import streamlit as st
from pathlib import Path

from app.chatbot import CustomerServiceBot
from app.config import KNOWLEDGE_BASE_PATH
from app.multilingual import LANG_NAMES
from app.dynamic_update import update_knowledge_base
from app.vector_store import LocalVectorStore
from app.config import PERSIST_DIR

st.set_page_config(page_title="Gen AI Customer Service Bot", layout="wide")
st.title("Real-Time Gen AI Customer Service Bot")

if "bot" not in st.session_state:
    st.session_state.bot = CustomerServiceBot()

bot = st.session_state.bot

with st.sidebar:
    st.header("Modes")
    mode = st.selectbox(
        "Choose chatbot mode",
        ["customer", "multimodal", "medical", "research"],
        format_func=lambda x: {
            "customer": "Customer Service",
            "multimodal": "Multimodal (Text + Image)",
            "medical": "Medical Q&A",
            "research": "Research Expert",
        }[x],
    )

    user_language = st.selectbox(
        "Output language",
        ["auto", "en", "hi", "es", "fr", "te"],
        format_func=lambda x: "Auto detect" if x == "auto" else LANG_NAMES.get(x, x),
    )

    st.caption("Task 1: Dynamic Knowledge Base Expansion")
    if st.button("Refresh Knowledge Base"):
        result = bot.refresh_knowledge_base([KNOWLEDGE_BASE_PATH])
        st.success(f"Updated knowledge base. Added {result.get('added', 0)} documents.")

    st.caption("Current environment")
    st.write("Gemini API key loaded:" , "Yes" if st.secrets.get("GEMINI_API_KEY", "") or True else "No")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Enter your message", height=140, placeholder="Ask a support question, a medical question, or a research query...")
    uploaded_image = None
    if mode == "multimodal":
        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if st.button("Send"):
        img_path = None
        if uploaded_image:
            img_path = Path(st.session_state.get("temp_img", "uploaded_image.png"))
            img_path = Path("/tmp") / uploaded_image.name
            img_path.write_bytes(uploaded_image.getvalue())

        result = bot.respond(
            user_input,
            mode=mode,
            image_path=str(img_path) if img_path else None,
            user_language=None if user_language == "auto" else user_language,
        )

        st.subheader("Response")
        st.write(result["response"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", result["sentiment"]["label"])
        c2.metric("Confidence", f"{result['sentiment']['score']:.2f}")
        c3.metric("Language", result["language"])

with col2:
    st.subheader("Knowledge Base Preview")
    preview = bot.customer_store.search("support policy", top_k=3)
    for item in preview:
        st.markdown(f"**{item['title']}**")
        st.caption(item["content"][:180] + ("..." if len(item["content"]) > 180 else ""))

st.divider()
st.caption("Built with NLP, TF-IDF retrieval, sentiment analysis, multilingual support, multimodal analysis, and optional Gemini integration.")
