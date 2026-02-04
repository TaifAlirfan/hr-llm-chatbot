import sqlite3
from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
DB_PATH = BASE_DIR / "db" / "hr.db"

TABLE_NAME = "employees"


def load_csv() -> pd.DataFrame:
    """Load HR dataset from CSV."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    return df


def create_database(df: pd.DataFrame) -> None:
    """Create SQLite database and store HR data."""
    DB_PATH.parent.mkdir(exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    print(f"Database created at {DB_PATH}")
    print(f"Table '{TABLE_NAME}' loaded with {len(df)} records.")


def preview_sql() -> None:
    """Run a quick SQL sanity check."""
    with sqlite3.connect(DB_PATH) as conn:
        query = f"""
        SELECT Department, COUNT(*) as count
        FROM {TABLE_NAME}
        GROUP BY Department
        ORDER BY count DESC
        """
        result = pd.read_sql(query, conn)
        print(result)


if __name__ == "__main__":
    df = load_csv()
    create_database(df)
    preview_sql()
