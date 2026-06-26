import pandas as pd

# Load dataset
df = pd.read_csv(
    "Amazon_DATA/train.csv",
    header=None,
    names=["label", "title", "review_text"]
)

# Convert labels
df["sentiment"] = df["label"].map({
    1: "negative",
    2: "positive"
})

print(f"Loaded {len(df)} reviews from Amazon_DATA/train.csv")

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

unique_reviews = df["review_text"].nunique()
total_reviews = len(df)

print(
    f"\nUnique reviews: {unique_reviews} out of {total_reviews} "
    f"({unique_reviews/total_reviews*100:.1f}% unique)"
)