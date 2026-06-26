"""
Step 8a: FastAPI Backend
Exposes the classifier, topic model, and RAG as REST API endpoints.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Customer Feedback Intelligence API")

# ── Load models on startup ───────────────────────────────────────────────────
print("Loading models...")
vectorizer = joblib.load("models/baseline/tfidf_vectorizer.joblib")
classifier = joblib.load("models/baseline/logreg_classifier.joblib")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_collection("reviews")
print("Models loaded.")

# ── Request/Response schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

class QueryRequest(BaseModel):
    question: str
    sentiment_filter: str | None = None

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_sentiment(req: PredictRequest):
    """Predict sentiment of a single review."""
    vec = vectorizer.transform([req.text])
    pred = classifier.predict(vec)[0]
    prob = classifier.predict_proba(vec)[0]
    return {
        "text": req.text,
        "sentiment": "positive" if pred == 1 else "negative",
        "confidence": round(float(max(prob)), 3)
    }


@app.post("/query")
def rag_query(req: QueryRequest):
    """Answer a natural language question using RAG over reviews."""
    question_embedding = embedder.encode([req.question]).tolist()

    where = None
    if req.sentiment_filter in ["positive", "negative"]:
        where = {"sentiment": req.sentiment_filter}

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=6,
        where=where
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not docs:
        return {"answer": "No relevant reviews found.", "sources": []}

    context = "\n".join(
        f"[{i+1}] ({meta['sentiment']}) {doc}"
        for i, (doc, meta) in enumerate(zip(docs, metadatas))
    )

    RAG_PROMPT = """You are a product analyst answering questions using real customer reviews.
Use ONLY the reviews below. Cite review numbers like [1], [2].
If reviews don't contain enough info, say so — do NOT make anything up.

Reviews:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{
            "role": "user",
            "content": RAG_PROMPT.format(context=context, question=req.question)
        }]
    )

    return {
        "answer": response["message"]["content"].strip(),
        "sources": [
            {"text": doc[:150], "sentiment": meta["sentiment"]}
            for doc, meta in zip(docs, metadatas)
        ]
    }


@app.get("/stats")
def get_stats():
    """Return dataset statistics for the dashboard."""
    df = pd.read_csv(
        "Amazon_DATA/train.csv",
        header=None,
        names=["label", "title", "review_text"]
    )
    df = df.head(3000)
    df["sentiment"] = df["label"].map({1: "negative", 2: "positive"})

    topics_df = pd.read_csv("data/processed/reviews_with_topics.csv")

    return {
        "total_reviews": len(df),
        "sentiment_distribution": df["sentiment"].value_counts().to_dict(),
        "topic_distribution": topics_df["topic_label"].value_counts().to_dict(),
        "baseline_accuracy": 0.8067,
        "transformer_accuracy": 0.8867,
    }