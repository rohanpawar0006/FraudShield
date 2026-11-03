"""Unit test verifying SQL schema parity between docker/init.sql, PostgreSQL, and SQLite statements."""

import re
from pathlib import Path

from src.common.db import DatabaseManager


def parse_columns_from_create_table(sql: str, table_name: str) -> list[str]:
    """Parse column names from a CREATE TABLE statement in SQL."""
    pattern = rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{table_name}\s*\((.*?)\);"
    match = re.search(pattern, sql, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"Table {table_name} not found in SQL")

    body = match.group(1)
    columns = []
    for line in body.split("\n"):
        line = line.strip().rstrip(",")
        if not line or line.startswith("--") or line.upper().startswith("CONSTRAINT") or line.upper().startswith("FOREIGN KEY") or line.upper().startswith("PRIMARY KEY"):
            continue
        col_name = line.split()[0].replace('"', "").replace("`", "")
        if col_name and not col_name.upper().startswith("INDEX"):
            columns.append(col_name)
    return columns


def extract_insert_columns(sql_snippet: str) -> list[str]:
    """Extract column names from an INSERT INTO statement."""
    match = re.search(r"INSERT\s+(?:OR\s+IGNORE\s+)?INTO\s+\w+\s*\((.*?)\)", sql_snippet, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"INSERT columns not found in {sql_snippet}")
    raw_cols = match.group(1).split(",")
    return [c.strip() for c in raw_cols]


def test_schema_parity_init_sql_vs_db_manager():
    """Verify that docker/init.sql matches DatabaseManager insert statements column-for-column."""
    init_sql_path = Path("docker/init.sql")
    assert init_sql_path.exists(), "docker/init.sql must exist"
    init_sql = init_sql_path.read_text(encoding="utf-8")

    # 1. Parse docker/init.sql tables
    expected_tx_cols = parse_columns_from_create_table(init_sql, "transactions")
    expected_score_cols = parse_columns_from_create_table(init_sql, "scores")

    # Exclude auto-increment id from scores INSERT expectation
    expected_score_insert_cols = [c for c in expected_score_cols if c != "id"]

    # 2. Check DatabaseManager source code for SQL statements
    db_file = Path("src/common/db.py").read_text(encoding="utf-8")

    # Transactions inserts
    pg_tx_match = re.search(r"INSERT\s+INTO\s+transactions\s*\((.*?)\)", db_file, re.IGNORECASE)
    assert pg_tx_match, "PostgreSQL transactions INSERT statement not found"
    pg_tx_cols = [c.strip() for c in pg_tx_match.group(1).split(",")]

    sqlite_tx_match = re.search(r"INSERT\s+OR\s+IGNORE\s+INTO\s+transactions\s*\((.*?)\)", db_file, re.IGNORECASE)
    assert sqlite_tx_match, "SQLite transactions INSERT statement not found"
    sqlite_tx_cols = [c.strip() for c in sqlite_tx_match.group(1).split(",")]

    assert pg_tx_cols == expected_tx_cols, f"PostgreSQL tx columns {pg_tx_cols} != init.sql {expected_tx_cols}"
    assert sqlite_tx_cols == expected_tx_cols, f"SQLite tx columns {sqlite_tx_cols} != init.sql {expected_tx_cols}"
    assert pg_tx_cols == sqlite_tx_cols, "PostgreSQL and SQLite transactions columns must be identical"

    # Scores inserts
    pg_score_match = re.search(r"INSERT\s+INTO\s+scores\s*\(\s*(txn_id.*?)\s*\)", db_file, re.IGNORECASE | re.DOTALL)
    assert pg_score_match, "PostgreSQL scores INSERT statement not found"
    pg_score_cols = [c.strip() for c in pg_score_match.group(1).split(",")]

    sqlite_score_match = re.search(r"INSERT\s+INTO\s+scores\s*\(\s*(txn_id.*?)\s*\)", db_file, re.IGNORECASE | re.DOTALL)
    assert sqlite_score_match, "SQLite scores INSERT statement not found"
    sqlite_score_cols = [c.strip() for c in sqlite_score_match.group(1).split(",")]

    assert pg_score_cols == expected_score_insert_cols, f"PostgreSQL score cols {pg_score_cols} != init.sql {expected_score_insert_cols}"
    assert sqlite_score_cols == expected_score_insert_cols, f"SQLite score cols {sqlite_score_cols} != init.sql {expected_score_insert_cols}"
    assert pg_score_cols == sqlite_score_cols, "PostgreSQL and SQLite scores columns must be identical"

    # 3. Check Cases and Labels insert statements in src/feedback/service.py
    feedback_file = Path("src/feedback/service.py").read_text(encoding="utf-8")
    expected_case_cols = parse_columns_from_create_table(init_sql, "cases")
    expected_label_cols = parse_columns_from_create_table(init_sql, "labels")
    expected_label_insert_cols = [c for c in expected_label_cols if c not in ("id", "labeled_at")]

    # Cases
    pg_case_match = re.search(r"INSERT\s+INTO\s+cases\s*\((.*?)\)", feedback_file, re.IGNORECASE)
    assert pg_case_match, "PostgreSQL cases INSERT statement not found"
    pg_case_cols = [c.strip() for c in pg_case_match.group(1).split(",")]

    sqlite_case_match = re.search(r"INSERT\s+INTO\s+cases\s*\((.*?)\)", feedback_file, re.IGNORECASE)
    assert sqlite_case_match, "SQLite cases INSERT statement not found"
    sqlite_case_cols = [c.strip() for c in sqlite_case_match.group(1).split(",")]

    assert pg_case_cols == expected_case_cols, f"PG case cols {pg_case_cols} != init.sql {expected_case_cols}"
    assert sqlite_case_cols == expected_case_cols, f"SQLite case cols {sqlite_case_cols} != init.sql {expected_case_cols}"

    # Labels
    pg_label_match = re.search(r"INSERT\s+INTO\s+labels\s*\((.*?)\)", feedback_file, re.IGNORECASE)
    assert pg_label_match, "PostgreSQL labels INSERT statement not found"
    pg_label_cols = [c.strip() for c in pg_label_match.group(1).split(",")]

    sqlite_label_match = re.search(r"INSERT\s+INTO\s+labels\s*\((.*?)\)", feedback_file, re.IGNORECASE)
    assert sqlite_label_match, "SQLite labels INSERT statement not found"
    sqlite_label_cols = [c.strip() for c in sqlite_label_match.group(1).split(",")]

    assert pg_label_cols == expected_label_insert_cols, f"PG label cols {pg_label_cols} != init.sql {expected_label_insert_cols}"
    assert sqlite_label_cols == expected_label_insert_cols, f"SQLite label cols {sqlite_label_cols} != init.sql {expected_label_insert_cols}"


def test_sqlite_runtime_table_schema(tmp_path):
    """Verify runtime SQLite table columns against docker/init.sql."""
    import sqlite3
    db_file = tmp_path / "schema_check.db"
    manager = DatabaseManager()
    manager._init_sqlite_schema(db_file)

    init_sql = Path("docker/init.sql").read_text(encoding="utf-8")
    expected_tx_cols = parse_columns_from_create_table(init_sql, "transactions")
    expected_score_cols = parse_columns_from_create_table(init_sql, "scores")
    expected_case_cols = parse_columns_from_create_table(init_sql, "cases")
    expected_label_cols = parse_columns_from_create_table(init_sql, "labels")

    with sqlite3.connect(str(db_file)) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(transactions);")
        tx_info = [row[1] for row in cur.fetchall()]
        assert tx_info == expected_tx_cols

        cur.execute("PRAGMA table_info(scores);")
        score_info = [row[1] for row in cur.fetchall()]
        assert score_info == expected_score_cols

        cur.execute("PRAGMA table_info(cases);")
        case_info = [row[1] for row in cur.fetchall()]
        assert case_info == expected_case_cols

        cur.execute("PRAGMA table_info(labels);")
        label_info = [row[1] for row in cur.fetchall()]
        assert label_info == expected_label_cols

