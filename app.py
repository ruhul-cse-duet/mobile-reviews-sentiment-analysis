import logging
import os
import time
import warnings
from pathlib import Path

import streamlit as st

from main import predict_sentiment

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = logging.getLogger(__name__)


def load_css():
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header():
    st.markdown(
        """
        <section class="hero">
            <div class="hero__text">
                <p class="eyebrow">Realtime insights · Cross-device ready</p>
                <h1>Global Mobile Reviews Sentiment</h1>
                <p class="subtitle">
                    Transform raw customer feedback into clear sentiment signals.
                    Drop any review and instantly see whether it is praising, neutral,
                    or highlighting problems — optimized for desktops, tablets, and phones.
                </p>
                <div class="hero__chips">
                    <span class="chip">Hybrid ONNX</span>
                    <span class="chip">SentenceTransformer</span>
                    <span class="chip">FastAPI + Streamlit</span>
                    <span class="chip">Responsive UI</span>
                </div>
            </div>
            <div class="hero__panel">
                <div class="panel-card">
                    <span class="label">Latency</span>
                    <span class="value">&lt; 1s</span>
                    <small>Single CPU prediction</small>
                </div>
                <div class="panel-card">
                    <span class="label">Confidence</span>
                    <span class="value">0–1</span>
                    <small>Calibrated softmax</small>
                </div>
                <div class="panel-card">
                    <span class="label">Coverage</span>
                    <span class="value">3 classes</span>
                    <small>Positive · Neutral · Negative</small>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metrics():
    st.markdown(
        """
        <section class="metrics">
            <div class="metric">
                <p class="metric__label">Encoder</p>
                <p class="metric__value">MiniLM-L6-v2</p>
                <p class="metric__hint">SentenceTransformer embeddings</p>
            </div>
            <div class="metric">
                <p class="metric__label">Serving</p>
                <p class="metric__value">FastAPI</p>
                <p class="metric__hint">/predict JSON endpoint</p>
            </div>
            <div class="metric">
                <p class="metric__label">UI</p>
                <p class="metric__value">Streamlit</p>
                <p class="metric__hint">One-click experiences</p>
            </div>
            <div class="metric">
                <p class="metric__label">Deployment</p>
                <p class="metric__value">Docker</p>
                <p class="metric__hint">Container ready build</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def set_example_text(text: str):
    st.session_state["review_text_buffer"] = text
    st.rerun()


def render_examples():
    examples = [
        "Battery lasts all weekend and the camera is flagship level.",
        "It is okay, but the screen scratches easily and nothing stands out.",
        "Laggy performance, weak signal, and the speaker cracked on day two.",
    ]
    st.markdown('<p class="eyebrow">Quick examples</p>', unsafe_allow_html=True)
    cols = st.columns(len(examples))
    for text, col in zip(examples, cols):
        with col:
            st.button(
                text[:28] + "...",
                key=f"example-{text[:10]}",
                on_click=set_example_text,
                args=(text,),
            )


def render_prediction(result, elapsed_ms=None, target=None):
    label = result.get("label", "Unknown")
    confidence = result.get("confidence", 0.0)

    color = {
        "Positive": "#16a34a",
        "Neutral": "#2563eb",
        "Negative": "#dc2626",
    }.get(label, "#6b21a8")

    target = target or st
    target.markdown(
        f"""
        <div class="prediction-card">
            <p class="eyebrow">Sentiment</p>
            <h3 style="color:{color};">{label}</h3>
            <p class="confidence">Confidence: {confidence:.3f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if elapsed_ms is not None:
        target.caption(f"Inference time: {elapsed_ms:.1f} ms on this device.")


def render_pipeline_card():
    st.markdown(
        """
        <section class="pipeline card">
            <h3>🧠 Pipeline overview</h3>
            <ol>
                <li><strong>Clean</strong>: Remove HTML, URLs, emojis, digits, and tidy whitespace.</li>
                <li><strong>Embed</strong>: Convert text into dense vectors with MiniLM-L6-v2.</li>
                <li><strong>Infer</strong>: Run the hybrid ONNX classifier for logits.</li>
                <li><strong>Calibrate</strong>: Apply softmax to surface sentiment confidence.</li>
            </ol>
            <p class="tip">Optimized to look sharp on mobile, tablets, ultrawide monitors, and embedded iframes.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Mobile Reviews Sentiment Classification",
        page_icon="📱",
        layout="centered",
    )
    load_css()

    if "review_text" not in st.session_state:
        st.session_state["review_text"] = ""
    if "review_text_buffer" not in st.session_state:
        st.session_state["review_text_buffer"] = None
    if st.session_state.get("review_text_buffer"):
        st.session_state["review_text"] = st.session_state["review_text_buffer"]
        st.session_state["review_text_buffer"] = None

    render_header()
    render_metrics()

    col_input, col_output = st.columns([0.58, 0.42], gap="large")

    with col_input:
        st.markdown('<div class="card glass-card">', unsafe_allow_html=True)
        st.subheader("✏️ Paste a mobile review")
        review_text = st.text_area(
            "",
            key="review_text",
            placeholder="Example: The battery life is amazing but the camera is just okay.",
            height=220,
            label_visibility="collapsed",
        )
        analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="card assist-card">', unsafe_allow_html=True)
        render_examples()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_output:
        result_placeholder = st.container()
        result_placeholder.markdown(
            """
            <div class="prediction-card placeholder">
                <p class="eyebrow">Sentiment</p>
                <h3>Awaiting input</h3>
                <p class="confidence">Enter a review to see the prediction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyze_btn:
        if not review_text.strip():
            st.warning("Please enter a review before submitting.")
        else:
            with st.spinner("Analyzing sentiment..."):
                start = time.perf_counter()
                result = predict_sentiment(review_text)
                elapsed = (time.perf_counter() - start) * 1000
                logger.info("Prediction completed in %.2f ms", elapsed)
            result_placeholder.empty()
            render_prediction(result, elapsed_ms=elapsed, target=result_placeholder)

    render_pipeline_card()


if __name__ == "__main__":
    main()
