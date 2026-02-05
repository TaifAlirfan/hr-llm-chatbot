import re
import sqlite3
from typing import List
import pandas as pd
from .config import DB_PATH, TABLE_NAME

ALLOWED_COLUMNS: List[str] = [
    "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField", "EmployeeCount",
    "EmployeeNumber", "EnvironmentSatisfaction", "Gender", "HourlyRate",
    "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
    "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "Over18", "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StandardHours", "StockOptionLevel",
    "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

SCHEMA_HINT = f"""
SQLite table: {TABLE_NAME}
Columns: {", ".join(ALLOWED_COLUMNS)}

Rules:
- Only generate SELECT queries.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, PRAGMA, ATTACH, DETACH, CREATE.
- Prefer aggregations (COUNT, AVG, MIN, MAX) when asked for summaries.
- Default LIMIT 50 unless the user asks for full output.
"""

FORBIDDEN_SQL = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|PRAGMA|ATTACH|DETACH|CREATE)\b"

def is_safe_sql(query: str) -> bool:
    q = query.strip().strip(";")
    if not q.lower().startswith("select"):
        return False
    if re.search(FORBIDDEN_SQL, q, flags=re.IGNORECASE):
        return False
    return True


def run_sql(query: str) -> pd.DataFrame:
    if not is_safe_sql(query):
        raise ValueError("Unsafe SQL detected. Only SELECT queries are allowed.")

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(query, conn)


if __name__ == "__main__":
    test_query = f"""
    SELECT Department, COUNT(*) as count
    FROM {TABLE_NAME}
    GROUP BY Department
    ORDER BY count DESC
    """
    print(run_sql(test_query))
