"""
Step 7: RAG (Retrieval Augmented Generation)
Lets anyone ask natural language questions over the reviews
and get back grounded, cited answers.
"""

import pandas as pd
import chromadb
import ollama
import os
from sentence_transformers import SentenceTransformer


# ── Build the vector index ──────────────────────────────────────────────────

def build_index():
    print("Loading reviews...")
    df = pd.read_csv(
        "Amazon_DATA/train.csv",
        header=None,
        names=["label", "title", "review_text"]
    )
    df = df.head(500)   # 500 reviews for portfolio demo — scale up later
    df = df.dropna(subset=["review_text"])
    df["review_text"] = df["review_text"].astype(str)
    df["sentiment"] = df["label"].map({1: "negative", 2: "positive"})
    df = df.reset_index(drop=True)

    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding reviews... (takes ~30 seconds)")
    embeddings = embedder.encode(df["review_text"].tolist(), show_progress_bar=True)

    print("Building vector index...")
    os.makedirs("data/chroma_db", exist_ok=True)
    client = chromadb.PersistentClient(path="data/chroma_db")

    # Delete existing collection if rebuilding
    try:
        client.delete_collection("reviews")
    except Exception:
        pass

    collection = client.create_collection("reviews")

    collection.add(
        ids=[str(i) for i in df.index],
        embeddings=embeddings.tolist(),
        documents=df["review_text"].tolist(),
        metadatas=[
            {
                "sentiment": row["sentiment"],
                "label": int(row["label"])
            }
            for _, row in df.iterrows()
        ]
    )

    print(f"Index built with {len(df)} reviews.")
    return collection, embedder


# ── Query the index ─────────────────────────────────────────────────────────

RAG_PROMPT = """You are a product analyst answering questions using real customer reviews.
Use ONLY the reviews provided below. Cite review numbers like [1], [2] in your answer.
If the reviews don't contain enough information, say so — do NOT make anything up.

Customer Reviews:
{context}

Question: {question}

Answer (cite review numbers):"""


def query(question: str, collection, embedder, sentiment_filter: str = None, top_k: int = 6):
    # Embed the question
    question_embedding = embedder.encode([question]).tolist()

    # Build filter if provided
    where = None
    if sentiment_filter:
        where = {"sentiment": sentiment_filter}

    # Retrieve most relevant reviews
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
        where=where
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not docs:
        return {
            "answer": "No relevant reviews found for this query.",
            "sources": []
        }

    # Build context string with numbered citations
    context = "\n".join(
        f"[{i+1}] ({meta['sentiment']}) {doc}"
        for i, (doc, meta) in enumerate(zip(docs, metadatas))
    )

    # Ask local LLM to answer using those reviews
    response = ollama.chat(
        model="llama3.2",
        messages=[{
            "role": "user",
            "content": RAG_PROMPT.format(context=context, question=question)
        }]
    )

    answer = response["message"]["content"].strip()

    return {
        "answer": answer,
        "sources": [
            {"text": doc[:150], "sentiment": meta["sentiment"]}
            for doc, meta in zip(docs, metadatas)
        ]
    }


# ── Main: build index then run sample queries ────────────────────────────────

def main():
    # Build index
    collection, embedder = build_index()

    # Sample questions to demonstrate RAG
    questions = [
        "What are customers complaining about most?",
        "What do customers love about their purchases?",
        "Which products have quality issues?",
    ]

    print("\n========== RAG QUERY RESULTS ==========\n")

    for question in questions:
        print(f"Q: {question}")
        print("-" * 50)
        result = query(question, collection, embedder)
        print(f"A: {result['answer']}")
        print(f"\nSources used:")
        for i, src in enumerate(result["sources"]):
            print(f"  [{i+1}] ({src['sentiment']}) {src['text']}...")
        print("\n" + "=" * 60 + "\n")

    # Interactive mode
    print("========== INTERACTIVE MODE ==========")
    print("Ask your own questions (type 'quit' to exit)\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            break
        if not question:
            continue

        sentiment = input("Filter by sentiment? (positive/negative/leave blank): ").strip()
        sentiment_filter = sentiment if sentiment in ["positive", "negative"] else None

        result = query(question, collection, embedder, sentiment_filter=sentiment_filter)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nBased on {len(result['sources'])} reviews:")
        for i, src in enumerate(result["sources"]):
            print(f"  [{i+1}] ({src['sentiment']}) {src['text']}...")
        print()


if __name__ == "__main__":
    main()