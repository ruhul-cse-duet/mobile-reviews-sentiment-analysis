import streamlit as st
from pathlib import Path
import time
import logging
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings('ignore')
import os
from main import predict_sentiment


def load_css():
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header():
    left, right = st.columns([0.6, 0.4])
    with left:

        st.markdown(
            "Analyze customer sentiment for **mobile phones** in seconds. "
            "Paste a review and get an instant prediction powered by a "
            "**Hybrid ONNX + SentenceTransformer** model."
        )
    with right:
        st.metric("Model Type", "Hybrid ONNX")
        st.metric("Task", "Sentiment Classification")


# def render_examples():
#     with st.expander("Need inspiration? Try some example reviews"):
#         examples = {
#             "Very Positive": "This phone is amazing! The battery lasts all day and the camera is crystal clear.",
#             "Neutral": "The phone is okay. Performance is fine but nothing special compared to my old one.",
#             "Negative": "Terrible experience. It keeps lagging and the battery drains in a few hours.",
#         }
#         cols = st.columns(len(examples))
#         for (label, text), col in zip(examples.items(), cols):
#             with col:
#                 if st.button(label):
#                     st.session_state["review_text"] = text


def render_prediction(result):
    label = result.get("label", "Unknown")
    confidence = result.get("confidence", 0.0)

    color = {
        "Positive": "#2e7d32",
        "Neutral": "#546e7a",
        "Negative": "#c62828",
    }.get(label, "#1976d2")

    st.markdown("### 🔍 Prediction")
    st.markdown(
        f"""
        <div style="padding: 1.2rem; border-radius: 0.75rem; border: 1px solid #e0e0e0;">
            <span style="font-size: 1.1rem; color: #555;">Sentiment</span><br>
            <span style="font-size: 1.8rem; font-weight: 700; color: {color};">{label}</span><br>
            <span style="font-size: 0.95rem; color: #777;">Confidence: {confidence:.3f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.title("📱 Global Mobile Reviews Sentiment")
    st.set_page_config(
        page_title="Mobile Reviews Sentiment Classification",
        page_icon="📱",
        layout="centered",
    )
    load_css()

    if "review_text" not in st.session_state:
        st.session_state["review_text"] = ""

    render_header()
    st.markdown("---")

    col_input, col_info = st.columns([0.65, 0.35])

    with col_input:
        st.subheader("✏️ Enter a mobile review")
        review_text = st.text_area(
            "",
            key="review_text",
            placeholder="Example: The battery life is amazing but the camera is just okay.",
            height=200,
        )
        analyze_btn = st.button("Analyze Sentiment", type="primary")

    with col_info:
        st.subheader("ℹ️ How it works")
        st.write(
            "- Text is cleaned (URLs, emojis, extra spaces removed).\n"
            "- A **SentenceTransformer** encodes the review.\n"
            "- A **hybrid ONNX model** predicts Negative / Neutral / Positive."
        )
        #render_examples()

    if analyze_btn:
        if not review_text.strip():
            st.warning("Please enter a review before submitting.")
        else:
            with st.spinner("Analyzing sentiment..."):
                result = predict_sentiment(review_text)
            render_prediction(result)


if __name__ == "__main__":
    main()

# streamlit run app.py
