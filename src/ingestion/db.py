from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    Text,
    DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
import pandas as pd

# -----------------------------
# Database Setup
# -----------------------------
Base = declarative_base()

class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True)
    review_text = Column(Text, nullable=False)
    gold_sentiment = Column(String, nullable=True)
    predicted_sentiment = Column(String, nullable=True)
    sentiment_confidence = Column(Float, nullable=True)
    urgency = Column(String, nullable=True)
    topic_id = Column(Integer, nullable=True)
    topic_label = Column(String, nullable=True)
    feature_mentioned = Column(String, nullable=True)
    competitor_mentioned = Column(String, nullable=True)
    is_actionable = Column(Boolean, nullable=True)
    created_at = Column(DateTime, nullable=False)

# SQLite database
engine = create_engine("sqlite:///reviews.db")
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

# -----------------------------
# Load Amazon Dataset
# -----------------------------
df = pd.read_csv(
    "Amazon_DATA/train.csv",
    header=None,
    names=["label", "title", "review_text"]
)

# Convert labels
df["gold_sentiment"] = df["label"].map({
    1: "negative",
    2: "positive"
})

# Start with 3000 rows for testing
df = df.head(3000)

# -----------------------------
# Insert Data
# -----------------------------
reviews = []

for _, row in df.iterrows():
    reviews.append(
        Review(
            review_text=str(row["review_text"]),
            gold_sentiment=row["gold_sentiment"],
            created_at=datetime.now(timezone.utc)  # ← fixed here
        )
    )

session.bulk_save_objects(reviews)
session.commit()

print(f"Loaded {len(df)} reviews into the database.")