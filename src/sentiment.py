from pathlib import Path
import pandas as pd
from transformers import pipeline

ROOT = Path(__file__).resolve().parent.parent

RAW_CSV = ROOT / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "sentiment_sample.csv"

# A lightweight sentiment model that works well for demos
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"


def build_text_column(df: pd.DataFrame) -> pd.Series:
    """
    Feature engineering: create a pseudo-text field from structured columns.
    This satisfies the requirement even though dataset has no native text.
    """
    return (
        "Employee works as a " + df["JobRole"].astype(str)
        + " in the " + df["Department"].astype(str) + " department. "
        + "Job level: " + df["JobLevel"].astype(str) + ". "
        + "OverTime: " + df["OverTime"].astype(str) + ". "
        + "Attrition: " + df["Attrition"].astype(str) + "."
    )


def main(sample_size: int = 200):
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Dataset not found at: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)

    # Create engineered text
    df["profile_text"] = build_text_column(df)

    # Take a small sample for speed (professional and practical)
    sample = df.sample(n=min(sample_size, len(df)), random_state=42).copy()

    # Sentiment pipeline
    clf = pipeline("sentiment-analysis", model=MODEL_ID)

    # Run sentiment
    results = clf(sample["profile_text"].tolist(), truncation=True)

    sample["sentiment_label"] = [r["label"] for r in results]
    sample["sentiment_score"] = [r["score"] for r in results]

    # Save output
    sample.to_csv(OUT_FILE, index=False)

    # Quick summary
    print(f"Saved sentiment results to: {OUT_FILE}")
    print(sample["sentiment_label"].value_counts())
    print("\nTop 5 rows:")
    print(sample[["EmployeeNumber", "Department", "JobRole", "profile_text", "sentiment_label", "sentiment_score"]].head())


if __name__ == "__main__":
    main()
