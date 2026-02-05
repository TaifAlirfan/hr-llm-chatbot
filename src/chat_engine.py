import re
import pandas as pd
from typing import List, Dict

from .rag_sql import SCHEMA_HINT, run_sql
from .llm_clients import chat


# -------------------------
# System Prompts
# -------------------------
SQL_SYSTEM = f"""
You are an expert data analyst who writes SQLite SQL for the HR dataset.

{SCHEMA_HINT}

CRITICAL RULES:
- Table name is exactly: employees
- Attrition is TEXT with values 'Yes' or 'No' (NOT 1/0). Never use Attrition = 1 or Attrition = 0.
- Department is TEXT.
- MonthlyIncome, Age, JobSatisfaction are numeric columns.
- For rates/percentages: avoid integer division by using 1.0 * ... or 100.0 * ...
- Output ONLY ONE SQL SELECT query. No explanations. No markdown fences.
"""

ANSWER_SYSTEM = """
You are a professional HR consultant.

Rules:
- Provide ONLY the final answer to the user.
- Do NOT explain your reasoning.
- Do NOT describe SQL execution.
- Do NOT mention tables, queries, or internal steps.
- Keep the answer concise and professional.
- Use at most 2 short bullet points or 1 short paragraph.

IMPORTANT:
- If results are present, NEVER say "no matching records".
- If results are empty, say no matching records were found and suggest a better filter.
"""


# -------------------------
# Helpers: normalization / validation
# -------------------------
def _normalize_sql(sql: str) -> str:
    s = (sql or "").replace("```sql", "").replace("```", "").strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_first_select(sql: str) -> str:
    s = _normalize_sql(sql)
    s = re.sub(r"\bOnly\b", "", s, flags=re.IGNORECASE).strip()

    if ";" in s:
        s = s.split(";")[0].strip() + ";"

    idx = s.lower().find("select")
    if idx > 0:
        s = s[idx:]

    return s.strip()


def _contains_bad_attrition(sql: str) -> bool:
    s = _extract_first_select(sql).lower()
    return ("attrition = 1" in s) or ("attrition=1" in s) or ("attrition = 0" in s) or ("attrition=0" in s)


def is_valid_sql_for_hr(sql: str) -> bool:
    s = _extract_first_select(sql).lower()

    if not s.startswith("select"):
        return False

    if " from employees" not in s:
        return False

    if _contains_bad_attrition(s):
        return False

    bad_patterns = [
        r"select\s+query\b",
        r"select\s+sql\b",
        r"select\s+data\b",
        r"select\s+\*\s*limit\b",
    ]
    for p in bad_patterns:
        if re.search(p, s):
            return False

    return True


def ensure_limit(sql: str, default_limit: int = 50) -> str:
    s = _extract_first_select(sql)

    # Don't add LIMIT to count queries
    if re.search(r"select\s+count\s*\(", s, flags=re.IGNORECASE):
        return s.rstrip(";") + ";"

    if "limit" not in s.lower():
        s = s.rstrip(";") + f" LIMIT {default_limit};"
    return s


def is_count_question(q: str) -> bool:
    q = q.lower()
    keywords = ["how many", "count", "number of", "total"]
    return any(k in q for k in keywords)


def _match_known_sql(user_question: str) -> str | None:
    """
    Deterministic templates for high-importance queries (must be correct).
    """
    q = user_question.strip().lower()

    # Attrition rate by department
    if "attrition rate" in q and "department" in q:
        return """
        SELECT
          Department,
          ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2)
            AS AttritionRatePercent
        FROM employees
        GROUP BY Department
        ORDER BY AttritionRatePercent DESC;
        """.strip()

    # Average MonthlyIncome by department
    if ("average" in q or "avg" in q) and "monthlyincome" in q and "department" in q:
        return """
        SELECT
          Department,
          ROUND(AVG(MonthlyIncome), 2) AS AvgMonthlyIncome
        FROM employees
        GROUP BY Department
        ORDER BY AvgMonthlyIncome DESC;
        """.strip()

    return None


# -------------------------
# Core: SQL generation / repair
# -------------------------
def generate_sql(user_question: str, memory: List[Dict[str, str]], mode: str) -> str:
    # 1) Deterministic known queries first
    known = _match_known_sql(user_question)
    if known:
        return known

    # 2) Deterministic simple count
    if is_count_question(user_question):
        if not any(w in user_question.lower() for w in ["where", "above", "below", "greater", "less", "older", "younger", ">", "<", "="]):
            return "SELECT COUNT(*) AS employee_count FROM employees;"

    def _llm_sql(extra_instruction: str = "") -> str:
        messages = [{"role": "system", "content": SQL_SYSTEM}]

        if memory:
            messages.append({"role": "user", "content": "Conversation context (for follow-up questions):"})
            messages.append({"role": "user", "content": summarize_memory(memory)})

        if extra_instruction:
            messages.append({"role": "user", "content": extra_instruction})

        messages.append({"role": "user", "content": f"Question: {user_question}"})

        res = chat(messages, temperature=0.0, mode=mode).text
        res = ensure_limit(res)
        return res

    # First attempt
    sql = _llm_sql()

    # Guard against Attrition = 1/0
    if _contains_bad_attrition(sql):
        sql = _llm_sql(
            "Reminder: Attrition is TEXT ('Yes'/'No'), never 1/0. Fix and output ONE valid SELECT query."
        )

    if is_valid_sql_for_hr(sql):
        return sql

    # One strict retry
    sql_retry = _llm_sql(
        "Your previous SQL was invalid. You MUST query the SQLite table 'employees' "
        "and output ONLY ONE valid SELECT statement. Include 'FROM employees'. "
        "Attrition is TEXT ('Yes'/'No'). For rates use 100.0*SUM(CASE...)/COUNT(*). "
        "If the question asks 'how many', use SELECT COUNT(*)."
    )

    return sql_retry


def repair_sql(user_question: str, bad_sql: str, error_msg: str, memory, mode: str) -> str:
    messages = [{"role": "system", "content": SQL_SYSTEM}]
    messages.append({"role": "user", "content": "The SQL you wrote failed in SQLite. Fix it."})
    messages.append({"role": "user", "content": f"Question: {user_question}"})
    messages.append({"role": "user", "content": f"Bad SQL:\n{bad_sql}"})
    messages.append({"role": "user", "content": f"SQLite error:\n{error_msg}"})
    messages.append({"role": "user", "content": "Return ONLY ONE corrected SQL query."})

    fixed = chat(messages, temperature=0.0, mode=mode).text
    fixed = ensure_limit(fixed)

    # Guard again
    if _contains_bad_attrition(fixed):
        fixed = fixed.replace("Attrition = 1", "Attrition = 'Yes'").replace("Attrition=1", "Attrition='Yes'")
        fixed = fixed.replace("Attrition = 0", "Attrition = 'No'").replace("Attrition=0", "Attrition='No'")

    return fixed


# -------------------------
# Memory + Answer
# -------------------------
def summarize_memory(memory: List[Dict[str, str]], max_turns: int = 6) -> str:
    recent = memory[-max_turns:]
    lines = []
    for m in recent:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def answer_with_data(
    user_question: str,
    sql: str,
    df: pd.DataFrame,
    memory: List[Dict[str, str]],
    mode: str
) -> str:
    # ✅ Stop hallucinations: if empty, return deterministic message (no LLM)
    if df is None or df.empty:
        return "No matching records were found. Try refining the question (e.g., specify a department, a range, or a specific attribute)."

    table_preview = df.head(20).to_markdown(index=False)

    messages = [{"role": "system", "content": ANSWER_SYSTEM}]

    if memory:
        messages.append({"role": "user", "content": "Conversation context (for follow-up questions):"})
        messages.append({"role": "user", "content": summarize_memory(memory)})

    # ✅ Hard constraint: results exist
    messages.append({"role": "user", "content": "The query returned rows. Do NOT say 'no matching records'."})

    messages.append({"role": "user", "content": f"User question: {user_question}"})
    messages.append({"role": "user", "content": f"Result preview:\n{table_preview}"})

    return chat(messages, temperature=0.2, mode=mode).text.strip()


# -------------------------
# Orchestrator
# -------------------------
def ask_hr_bot(user_question: str, memory: List[Dict[str, str]], mode: str):
    sql = generate_sql(user_question, memory, mode)

    if not is_valid_sql_for_hr(sql):
        sql = ensure_limit(
            chat(
                [
                    {"role": "system", "content": SQL_SYSTEM},
                    {"role": "user", "content": "Write ONE valid SQLite SELECT query using FROM employees only."},
                    {"role": "user", "content": "Attrition is TEXT ('Yes'/'No'), never 1/0."},
                    {"role": "user", "content": "For rates use 100.0*SUM(CASE...)/COUNT(*)."},
                    {"role": "user", "content": f"Question: {user_question}"},
                ],
                temperature=0.0,
                mode=mode,
            ).text
        )

    try:
        df = run_sql(sql)
    except Exception as e:
        sql = repair_sql(user_question, sql, str(e), memory, mode)
        df = run_sql(sql)

    answer = answer_with_data(user_question, sql, df, memory, mode)
    return answer, sql, df
