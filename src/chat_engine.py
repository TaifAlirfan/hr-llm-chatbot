import re
import pandas as pd
from typing import List, Dict, Tuple

from .rag_sql import SCHEMA_HINT, run_sql
from .llm_clients import chat


SQL_SYSTEM = f"""
You are an expert data analyst who writes SQLite SQL.
{SCHEMA_HINT}

Output ONLY ONE SQL query.
No explanations. No markdown fences.
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

If the result table is empty, say no matching records were found and suggest a better filter.
"""



# -------------------------
# Helpers: normalization / validation
# -------------------------
def _normalize_sql(sql: str) -> str:
    """Remove markdown fences and extra whitespace."""
    s = (sql or "").replace("```sql", "").replace("```", "").strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_first_select(sql: str) -> str:
    """
    If the model outputs multiple statements or repeats itself,
    keep only the first SELECT...; statement.
    """
    s = _normalize_sql(sql)

    # Remove the hallucinated word "Only" (common in weak local outputs)
    s = re.sub(r"\bOnly\b", "", s, flags=re.IGNORECASE).strip()

    # Take only the first statement up to first semicolon
    if ";" in s:
        s = s.split(";")[0].strip() + ";"

    # Ensure it starts with SELECT
    idx = s.lower().find("select")
    if idx > 0:
        s = s[idx:]

    return s.strip()


def is_valid_sql_for_hr(sql: str) -> bool:
    """
    Strict validation to keep local models on track.
    Ensures SQL references the correct table and looks like a real query.
    """
    s = _extract_first_select(sql).lower()

    if not s.startswith("select"):
        return False

    if " from employees" not in s:
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
    """Add LIMIT if missing, but do NOT add LIMIT to pure COUNT queries."""
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


# -------------------------
# Core: SQL generation / repair
# -------------------------
def generate_sql(user_question: str, memory: List[Dict[str, str]], mode: str) -> str:
    # ✅ Hard rule: count questions should be deterministic and not rely on weak local models
    if is_count_question(user_question):
        # Generic count; if user also mentions a condition (e.g., above 39) LLM will still be used.
        # We'll only force raw count if it's a plain "how many employees are there" style.
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

    if is_valid_sql_for_hr(sql):
        return sql

    # One strict retry
    sql_retry = _llm_sql(
        "Your previous SQL was invalid. You MUST query the SQLite table 'employees' "
        "and output ONLY ONE valid SELECT statement. Include 'FROM employees'. "
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
    table_preview = df.head(20).to_markdown(index=False) if not df.empty else "(no rows)"

    messages = [{"role": "system", "content": ANSWER_SYSTEM}]

    if memory:
        messages.append({"role": "user", "content": "Conversation context (for follow-up questions):"})
        messages.append({"role": "user", "content": summarize_memory(memory)})

    messages.append({"role": "user", "content": f"User question: {user_question}"})
    messages.append({"role": "user", "content": f"SQL used:\n{sql}"})
    messages.append({"role": "user", "content": f"Result preview:\n{table_preview}"})

    return chat(messages, temperature=0.2, mode=mode).text.strip()


# -------------------------
# Orchestrator
# -------------------------
def ask_hr_bot(user_question: str, memory: List[Dict[str, str]], mode: str):
    sql = generate_sql(user_question, memory, mode)

    # If still invalid after retry, force one more regen with strictness
    if not is_valid_sql_for_hr(sql):
        sql = ensure_limit(
            chat(
                [
                    {"role": "system", "content": SQL_SYSTEM},
                    {"role": "user", "content": "Write ONE valid SQLite SELECT query using FROM employees only."},
                    {"role": "user", "content": "If the question asks 'how many', use SELECT COUNT(*)."},
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
