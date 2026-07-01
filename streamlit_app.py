"""
Customer Feedback Intelligence Platform
Standalone Streamlit app — no FastAPI needed
Works on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

st.set_page_config(
    page_title="Customer Feedback Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1a1d2e;
        border: 1px solid #2d3148;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: #888;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #ffffff;
        margin: 4px 0 0;
    }
    .metric-delta { font-size: 12px; margin: 4px 0 0; }
    .section-title {
        font-size: 13px;
        font-weight: 600;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 12px;
    }
    .tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin: 3px;
        background: #1e3a5f;
        color: #5b9ef7;
        border: 1px solid #2a5298;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("## 🧠 Customer Feedback Intelligence Platform")
st.caption("End-to-end NLP · Sentiment · Topic Modeling · RAG · DistilBERT · FastAPI · Streamlit")
st.divider()

# ── KPI Cards ────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown("""<div class='metric-card'>
        <p class='metric-label'>Reviews Processed</p>
        <p class='metric-value'>3,000</p>
        <p class='metric-delta' style='color:#5b9ef7;'>Amazon dataset</p>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown("""<div class='metric-card'>
        <p class='metric-label'>Baseline F1</p>
        <p class='metric-value'>80.7%</p>
        <p class='metric-delta' style='color:#888;'>TF-IDF + LogReg</p>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown("""<div class='metric-card'>
        <p class='metric-label'>DistilBERT F1</p>
        <p class='metric-value'>88.7%</p>
        <p class='metric-delta' style='color:#1baf7a;'>+8.0% improvement</p>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown("""<div class='metric-card'>
        <p class='metric-label'>Topics Found</p>
        <p class='metric-value'>8</p>
        <p class='metric-delta' style='color:#888;'>LDA clustering</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts ───────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.markdown("<p class='section-title'>Sentiment Distribution</p>", unsafe_allow_html=True)
    fig_pie = go.Figure(go.Pie(
        labels=["Positive", "Negative"],
        values=[47.1, 52.9],
        hole=0.6,
        marker=dict(colors=["#2a78d6", "#e34948"]),
        textinfo="label+percent",
        textfont=dict(size=13, color="white"),
    ))
    fig_pie.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=0),
        height=220, font=dict(color="white")
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with c2:
    st.markdown("<p class='section-title'>Model Performance Comparison</p>", unsafe_allow_html=True)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=["TF-IDF + LogReg", "DistilBERT (fine-tuned)"],
        x=[80.7, 88.7],
        orientation="h",
        marker=dict(color=["#2a78d6", "#1baf7a"]),
        text=["80.7%", "88.7%"],
        textposition="inside",
        textfont=dict(color="white", size=13),
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=0),
        height=220,
        xaxis=dict(range=[70, 95], gridcolor="#2d3148", color="#888"),
        yaxis=dict(color="#aaa"),
        font=dict(color="white")
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Topics ───────────────────────────────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    st.markdown("<p class='section-title'>Top Topics Discovered</p>", unsafe_allow_html=True)
    topics_df = pd.DataFrame({
        "Topic": ["Topic 1 (books)", "Positive experience",
                  "Topic 6 (products)", "Topic 2 (movies)", "Topic 5 (games)"],
        "Count": [943, 698, 694, 407, 132]
    })
    fig_topics = px.bar(
        topics_df, x="Count", y="Topic", orientation="h",
        color="Count",
        color_continuous_scale=["#1e3a5f", "#2a78d6", "#5b9ef7"],
        text="Count"
    )
    fig_topics.update_traces(textposition="inside", textfont=dict(color="white"))
    fig_topics.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=0),
        height=220,
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#2d3148", color="#888"),
        yaxis=dict(color="#aaa"),
        font=dict(color="white")
    )
    st.plotly_chart(fig_topics, use_container_width=True)

with c4:
    st.markdown("<p class='section-title'>LLM Extraction Results</p>", unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    e3, e4 = st.columns(2)
    with e1:
        st.markdown("""<div class='metric-card'>
            <p class='metric-label'>Features extracted</p>
            <p class='metric-value' style='font-size:22px;'>42</p>
        </div>""", unsafe_allow_html=True)
    with e2:
        st.markdown("""<div class='metric-card'>
            <p class='metric-label'>Competitors found</p>
            <p class='metric-value' style='font-size:22px;'>10</p>
        </div>""", unsafe_allow_html=True)
    with e3:
        st.markdown("""<div class='metric-card'>
            <p class='metric-label'>Actionable issues</p>
            <p class='metric-value' style='font-size:22px;'>4</p>
        </div>""", unsafe_allow_html=True)
    with e4:
        st.markdown("""<div class='metric-card'>
            <p class='metric-label'>Parse success</p>
            <p class='metric-value' style='font-size:22px;'>98%</p>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Sentiment Predictor ──────────────────────────────────────────────────────
st.markdown("## 🔍 Predict Sentiment")
st.caption("Uses the TF-IDF + Logistic Regression baseline model")

@st.cache_resource
def load_model():
    try:
        vectorizer = joblib.load("models/baseline/tfidf_vectorizer.joblib")
        clf = joblib.load("models/baseline/logreg_classifier.joblib")
        return vectorizer, clf
    except Exception:
        return None, None

vectorizer, clf = load_model()

user_review = st.text_area(
    "Paste a review here:",
    placeholder="e.g. This product is amazing, works perfectly!"
)

if st.button("Predict") and user_review:
    if vectorizer and clf:
        vec = vectorizer.transform([user_review])
        pred = clf.predict(vec)[0]
        prob = clf.predict_proba(vec)[0]
        confidence = round(float(max(prob)) * 100, 1)
        if pred == 1:
            st.success(f"✅ Positive — {confidence}% confidence")
        else:
            st.error(f"❌ Negative — {confidence}% confidence")
    else:
        st.warning("Model not found. Make sure models/baseline/ folder exists.")

st.divider()

# ── RAG Example ──────────────────────────────────────────────────────────────
# ── RAG Query ────────────────────────────────────────────────────────────────
st.markdown("## 💬 Ask a Question About Your Customers")
st.caption("Powered by ChromaDB + sentence-transformers + llama3.2 via Ollama")

@st.cache_resource
def load_rag():
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        client = chromadb.PersistentClient(path="data/chroma_db")
        collection = client.get_collection("reviews")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return collection, embedder
    except Exception as e:
        return None, None

collection, embedder = load_rag()

RAG_PROMPT = """You are a product analyst answering questions using real customer reviews.
Use ONLY the reviews below. Cite review numbers like [1], [2].
If reviews don't contain enough info, say so — do NOT make anything up.

Reviews:
{context}

Question: {question}

Answer:"""

question = st.text_input(
    "Your question:",
    placeholder="e.g. What are customers complaining about most?"
)

sentiment_filter = st.selectbox(
    "Filter by sentiment (optional):",
    ["None", "positive", "negative"]
)

if st.button("Ask") and question:
    if collection and embedder:
        with st.spinner("Searching reviews and generating answer..."):
            try:
                import ollama

                # Embed question
                question_embedding = embedder.encode([question]).tolist()

                # Filter
                where = None
                if sentiment_filter != "None":
                    where = {"sentiment": sentiment_filter}

                # Retrieve relevant reviews
                results = collection.query(
                    query_embeddings=question_embedding,
                    n_results=6,
                    where=where
                )

                docs = results["documents"][0]
                metadatas = results["metadatas"][0]

                if not docs:
                    st.warning("No relevant reviews found for this query.")
                else:
                    context = "\n".join(
                        f"[{i+1}] ({meta['sentiment']}) {doc}"
                        for i, (doc, meta) in enumerate(zip(docs, metadatas))
                    )

                    response = ollama.chat(
                        model="llama3.2",
                        messages=[{
                            "role": "user",
                            "content": RAG_PROMPT.format(
                                context=context,
                                question=question
                            )
                        }]
                    )

                    answer = response["message"]["content"].strip()

                    st.markdown("### Answer")
                    st.write(answer)

                    with st.expander("📚 Source Reviews Used"):
                        for i, (doc, meta) in enumerate(zip(docs, metadatas)):
                            badge = "🟢" if meta["sentiment"] == "positive" else "🔴"
                            st.write(f"{badge} [{i+1}] {doc[:200]}...")

            except Exception as e:
                st.error(f"Error: {str(e)} — Make sure Ollama is running (ollama serve)")
    else:
        st.warning("Vector database not found. Run src/rag/rag_engine.py first to build the index.")