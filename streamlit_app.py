"""
Step 8b: Streamlit Dashboard
Visual frontend for the Customer Feedback Intelligence Platform.
"""

import streamlit as st
import requests
import plotly.express as px
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Feedback Intelligence Platform")
st.caption("End-to-end NLP system — sentiment, topics, and RAG-powered Q&A over customer reviews")

# ── Fetch stats from API ─────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_stats():
    try:
        return requests.get(f"{API_URL}/stats").json()
    except Exception:
        return None

stats = fetch_stats()

# ── KPI Cards ────────────────────────────────────────────────────────────────
if stats:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{stats['total_reviews']:,}")
    col2.metric("Baseline Accuracy", f"{stats['baseline_accuracy']*100:.1f}%")
    col3.metric("DistilBERT Accuracy", f"{stats['transformer_accuracy']*100:.1f}%")
    col4.metric("Improvement", f"+{(stats['transformer_accuracy']-stats['baseline_accuracy'])*100:.1f}%")

st.divider()

# ── Charts ───────────────────────────────────────────────────────────────────
if stats:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_df = pd.DataFrame(
            list(stats["sentiment_distribution"].items()),
            columns=["Sentiment", "Count"]
        )
        fig = px.pie(
            sentiment_df, values="Count", names="Sentiment",
            color="Sentiment",
            color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Topics")
        topic_df = pd.DataFrame(
            list(stats["topic_distribution"].items()),
            columns=["Topic", "Count"]
        ).sort_values("Count", ascending=True).tail(8)
        fig = px.bar(
            topic_df, x="Count", y="Topic",
            orientation="h",
            color="Count",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Model Comparison ─────────────────────────────────────────────────────────
st.subheader("📈 Model Performance Comparison")
model_df = pd.DataFrame({
    "Model": ["TF-IDF + LogReg (Baseline)", "DistilBERT (Fine-tuned)"],
    "Accuracy": [0.8067, 0.8867],
    "F1 Score": [0.8067, 0.8867]
})
fig = px.bar(
    model_df, x="Model", y="Accuracy",
    color="Model",
    text="Accuracy",
    color_discrete_sequence=["#3498db", "#2ecc71"]
)
fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
fig.update_layout(yaxis_range=[0.7, 1.0], showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Sentiment Predictor ──────────────────────────────────────────────────────
st.subheader("🔍 Predict Sentiment")
user_review = st.text_area("Paste a review here:", placeholder="Type or paste a customer review...")

if st.button("Predict") and user_review:
    try:
        resp = requests.post(f"{API_URL}/predict", json={"text": user_review})
        result = resp.json()
        sentiment = result["sentiment"]
        confidence = result["confidence"]

        if sentiment == "positive":
            st.success(f"✅ Positive — {confidence*100:.1f}% confidence")
        else:
            st.error(f"❌ Negative — {confidence*100:.1f}% confidence")
    except Exception:
        st.warning("API not reachable. Make sure FastAPI is running.")

st.divider()

# ── RAG Query ────────────────────────────────────────────────────────────────
st.subheader("💬 Ask a Question About Your Customers")
question = st.text_input(
    "Question:",
    placeholder="e.g. What are customers complaining about most?"
)
sentiment_filter = st.selectbox(
    "Filter by sentiment (optional):",
    ["None", "positive", "negative"]
)

if st.button("Ask") and question:
    with st.spinner("Searching reviews and generating answer..."):
        try:
            payload = {
                "question": question,
                "sentiment_filter": None if sentiment_filter == "None" else sentiment_filter
            }
            resp = requests.post(f"{API_URL}/query", json=payload)
            result = resp.json()

            st.markdown("### Answer")
            st.write(result["answer"])

            with st.expander("📚 Source Reviews Used"):
                for i, src in enumerate(result["sources"]):
                    badge = "🟢" if src["sentiment"] == "positive" else "🔴"
                    st.write(f"{badge} [{i+1}] {src['text']}...")
        except Exception:
            st.warning("API not reachable. Make sure FastAPI is running.")