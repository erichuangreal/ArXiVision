"""Local SQLite database: one file, tables scoped by user_id.

Not one database file per user - the corpus (papers/processed_text/chroma_store)
is what's genuinely large and per-user, so those stay as per-user directories.
Metadata (accounts, collections, job status) is small and relational, so it
belongs in one file with a user_id column, like any ordinary multi-user app.
"""

import json
import secrets
import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path("data") / "app.db"

# Kept as plain constants (not imported from rag_class) so this module stays
# dependency-free - it only needs to know the shape of a settings row.
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_LANGUAGE_STYLE = "standard"


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _add_column_if_missing(conn, table, column, ddl):
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT,
                api_key TEXT UNIQUE NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        # Added after the table already existed for real accounts, so these
        # are migrated in rather than declared in the CREATE TABLE above.
        _add_column_if_missing(conn, "users", "model", f"model TEXT NOT NULL DEFAULT '{DEFAULT_MODEL}'")
        _add_column_if_missing(conn, "users", "temperature", f"temperature REAL NOT NULL DEFAULT {DEFAULT_TEMPERATURE}")
        _add_column_if_missing(
            conn, "users", "language_style", f"language_style TEXT NOT NULL DEFAULT '{DEFAULT_LANGUAGE_STYLE}'"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS collections (
                collection_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(user_id),
                question TEXT NOT NULL,
                papers_json TEXT NOT NULL,
                comparison_json TEXT,
                followups_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        # One row per (user, action, day); incremented and capped in
        # api.py so a leaked or shared key can't run up unbounded API spend.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                user_id TEXT NOT NULL REFERENCES users(user_id),
                action TEXT NOT NULL,
                day TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, action, day)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingest_jobs (
                job_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(user_id),
                topic TEXT NOT NULL,
                num_papers INTEGER NOT NULL,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                message TEXT,
                current_step INTEGER,
                total_steps INTEGER,
                error TEXT,
                started_at REAL,
                updated_at REAL
            )
        """)
        # One row per generated verification question, accumulated across
        # expeditions - a paper already covered isn't re-billed for a new
        # question on a later expedition.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS verification_questions (
                question_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(user_id),
                paper_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        # The latest verification ledger run, one row per user (not history -
        # each expedition re-runs scoring over the accumulated question set
        # and replaces this).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                user_id TEXT PRIMARY KEY REFERENCES users(user_id),
                results_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)


# --- users -------------------------------------------------------------

def create_user(email=None):
    user_id = uuid.uuid4().hex[:12]
    api_key = secrets.token_hex(24)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (user_id, email, api_key, created_at) VALUES (?, ?, ?, ?)",
            (user_id, email, api_key, time.time()),
        )
    return user_id, api_key


def get_user_by_api_key(api_key):
    with _connect() as conn:
        row = conn.execute(
            "SELECT user_id FROM users WHERE api_key = ?", (api_key,)
        ).fetchone()
    return row["user_id"] if row else None


def get_settings(user_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT model, temperature, language_style FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row is None:
        return None
    return {"model": row["model"], "temperature": row["temperature"], "language_style": row["language_style"]}


def update_settings(user_id, **fields):
    columns = {"model": "model", "temperature": "temperature", "language_style": "language_style"}
    sets, values = [], []
    for key, value in fields.items():
        sets.append(f"{columns[key]} = ?")
        values.append(value)
    values.append(user_id)

    with _connect() as conn:
        conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE user_id = ?", values)
    return get_settings(user_id)


# --- collections ---------------------------------------------------------

def _collection_row_to_dict(row):
    return {
        "collection_id": row["collection_id"],
        "question": row["question"],
        "papers": json.loads(row["papers_json"]),
        "comparison": json.loads(row["comparison_json"]) if row["comparison_json"] else None,
        "followups": json.loads(row["followups_json"]) if row["followups_json"] else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_collection(user_id, question, papers):
    collection_id = uuid.uuid4().hex[:8]
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO collections
               (collection_id, user_id, question, papers_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (collection_id, user_id, question, json.dumps(papers), now, now),
        )
    return get_collection(user_id, collection_id)


def get_collection(user_id, collection_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM collections WHERE collection_id = ? AND user_id = ?",
            (collection_id, user_id),
        ).fetchone()
    return _collection_row_to_dict(row) if row else None


def list_collections(user_id):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM collections WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [_collection_row_to_dict(row) for row in rows]


def update_collection(user_id, collection_id, **fields):
    if get_collection(user_id, collection_id) is None:
        return None
    columns = {"comparison": "comparison_json", "followups": "followups_json"}
    sets, values = [], []
    for key, value in fields.items():
        sets.append(f"{columns[key]} = ?")
        values.append(json.dumps(value))
    sets.append("updated_at = ?")
    values.append(time.time())
    values.extend([collection_id, user_id])

    with _connect() as conn:
        conn.execute(
            f"UPDATE collections SET {', '.join(sets)} WHERE collection_id = ? AND user_id = ?",
            values,
        )
    return get_collection(user_id, collection_id)


def add_paper_to_collection(user_id, collection_id, paper):
    collection = get_collection(user_id, collection_id)
    if collection is None:
        return None

    papers = [p for p in collection["papers"] if p["paper_id"] != paper["paper_id"]]
    papers.append(paper)

    with _connect() as conn:
        conn.execute(
            # Changing the paper set invalidates any existing comparison/
            # follow-ups, since they were computed over the old set.
            """UPDATE collections
               SET papers_json = ?, comparison_json = NULL, followups_json = NULL,
                   updated_at = ?
               WHERE collection_id = ? AND user_id = ?""",
            (json.dumps(papers), time.time(), collection_id, user_id),
        )
    return get_collection(user_id, collection_id)


def remove_paper_from_collection(user_id, collection_id, paper_id):
    collection = get_collection(user_id, collection_id)
    if collection is None:
        return None

    papers = [p for p in collection["papers"] if p["paper_id"] != paper_id]
    if len(papers) == len(collection["papers"]):
        return "not_found"  # paper_id wasn't in this collection

    with _connect() as conn:
        conn.execute(
            """UPDATE collections
               SET papers_json = ?, comparison_json = NULL, followups_json = NULL,
                   updated_at = ?
               WHERE collection_id = ? AND user_id = ?""",
            (json.dumps(papers), time.time(), collection_id, user_id),
        )
    return get_collection(user_id, collection_id)


# --- ingest jobs -----------------------------------------------------------

def _job_row_to_dict(row):
    return {
        "job_id": row["job_id"],
        "user_id": row["user_id"],
        "topic": row["topic"],
        "num_papers": row["num_papers"],
        "status": row["status"],
        "stage": row["stage"],
        "message": row["message"],
        "current": row["current_step"],
        "total": row["total_steps"],
        "error": row["error"],
        "started_at": row["started_at"],
        "updated_at": row["updated_at"],
    }


def create_job(user_id, job_id, topic, num_papers):
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO ingest_jobs
               (job_id, user_id, topic, num_papers, status, stage, message,
                current_step, total_steps, error, started_at, updated_at)
               VALUES (?, ?, ?, ?, 'running', 'queued', 'Waiting to start...', 0, ?, NULL, ?, ?)""",
            (job_id, user_id, topic, num_papers, num_papers, now, now),
        )
    return get_job(user_id, job_id)


def update_job(user_id, job_id, **fields):
    columns = {
        "status": "status", "stage": "stage", "message": "message",
        "current": "current_step", "total": "total_steps", "error": "error",
    }
    sets, values = [], []
    for key, value in fields.items():
        sets.append(f"{columns[key]} = ?")
        values.append(value)
    sets.append("updated_at = ?")
    values.append(time.time())
    values.extend([job_id, user_id])

    with _connect() as conn:
        conn.execute(
            f"UPDATE ingest_jobs SET {', '.join(sets)} WHERE job_id = ? AND user_id = ?",
            values,
        )
    return get_job(user_id, job_id)


def get_job(user_id, job_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM ingest_jobs WHERE job_id = ? AND user_id = ?",
            (job_id, user_id),
        ).fetchone()
    return _job_row_to_dict(row) if row else None


def list_jobs(user_id):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM ingest_jobs WHERE user_id = ? ORDER BY started_at DESC",
            (user_id,),
        ).fetchall()
    return [_job_row_to_dict(row) for row in rows]


def mark_interrupted_jobs():
    # A job still "running" when the process starts had its thread die with
    # the old process; it will never update again, so say so. stage is left
    # alone - it still names the real step the job was stuck in.
    with _connect() as conn:
        conn.execute(
            """UPDATE ingest_jobs SET status = 'failed',
               error = 'Interrupted by a server restart.',
               message = 'Interrupted by a server restart.'
               WHERE status = 'running'"""
        )


# --- verification questions & evaluation results ---------------------------

def verification_question_paper_ids(user_id):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT paper_id FROM verification_questions WHERE user_id = ?", (user_id,)
        ).fetchall()
    return {row["paper_id"] for row in rows}


def add_verification_questions(user_id, questions):
    now = time.time()
    with _connect() as conn:
        for question in questions:
            conn.execute(
                """INSERT INTO verification_questions
                   (question_id, user_id, paper_id, payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (uuid.uuid4().hex[:12], user_id, question["paper_id"], json.dumps(question), now),
            )


def list_verification_questions(user_id):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT payload_json FROM verification_questions WHERE user_id = ? ORDER BY created_at",
            (user_id,),
        ).fetchall()
    return [json.loads(row["payload_json"]) for row in rows]


def save_evaluation_results(user_id, results):
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO evaluation_results (user_id, results_json, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   results_json = excluded.results_json,
                   updated_at = excluded.updated_at""",
            (user_id, json.dumps(results), now),
        )


def get_evaluation_results(user_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT results_json, updated_at FROM evaluation_results WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row is None:
        return None
    result = json.loads(row["results_json"])
    result["updated_at"] = row["updated_at"]
    return result


# --- usage caps --------------------------------------------------------

def _today():
    return time.strftime("%Y-%m-%d", time.gmtime())


def usage_today(user_id, action):
    with _connect() as conn:
        row = conn.execute(
            "SELECT count FROM api_usage WHERE user_id = ? AND action = ? AND day = ?",
            (user_id, action, _today()),
        ).fetchone()
    return row["count"] if row else 0


def try_consume_usage(user_id, action, limit):
    # Atomic check-and-increment: returns True and records the use if under
    # limit, False (and records nothing) if already at it. One connection,
    # one transaction, so two concurrent requests can't both slip through.
    day = _today()
    with _connect() as conn:
        row = conn.execute(
            "SELECT count FROM api_usage WHERE user_id = ? AND action = ? AND day = ?",
            (user_id, action, day),
        ).fetchone()
        count = row["count"] if row else 0
        if count >= limit:
            return False
        if row:
            conn.execute(
                "UPDATE api_usage SET count = count + 1 WHERE user_id = ? AND action = ? AND day = ?",
                (user_id, action, day),
            )
        else:
            conn.execute(
                "INSERT INTO api_usage (user_id, action, day, count) VALUES (?, ?, ?, 1)",
                (user_id, action, day),
            )
    return True


init_db()
