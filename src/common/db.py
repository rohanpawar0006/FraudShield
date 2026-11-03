"""Database connectivity and asynchronous persistence layer for FraudShield.

Supports PostgreSQL with graceful SQLite local fallback when Docker/PostgreSQL is unavailable.
All database writes are executed off the critical scoring path using a background worker queue.
"""

import json
import queue
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import Json

from src.common.config import settings
from src.common.logging import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Manages database connection lifecycle and asynchronous off-path transaction/score persistence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.failed_writes_count: int = 0
        self.successful_writes_count: int = 0
        self._backend: str = "uninitialized"
        self._write_queue: queue.Queue = queue.Queue(maxsize=100000)
        self._running: bool = True

        self._initialize_backend()

        # Start background writer thread
        self._worker_thread = threading.Thread(
            target=self._process_write_queue, daemon=True
        )
        self._worker_thread.start()

    def _initialize_backend(self) -> None:
        """Attempt connection to PostgreSQL, falling back to SQLite if unreachable."""
        if not settings.USE_SQLITE_FALLBACK:
            try:
                conn = psycopg2.connect(
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    dbname=settings.POSTGRES_DB,
                    connect_timeout=2,
                )
                conn.close()
                self._backend = "postgresql"
                logger.info("Connected to PostgreSQL backend successfully")
                return
            except Exception as e:
                logger.warning(
                    "PostgreSQL unavailable (%s). Falling back to SQLite backend.",
                    e,
                )

        # Fallback: SQLite
        self._backend = "sqlite"
        db_path = Path(settings.SQLITE_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_sqlite_schema(db_path)
        logger.info(
            "Initialized SQLite storage fallback at %s", db_path.as_posix()
        )

    def _init_sqlite_schema(self, db_path: Path) -> None:
        """Initialize SQLite tables mirroring docker/init.sql with WAL performance optimizations."""
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    txn_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    dest_id TEXT NOT NULL,
                    amount REAL NOT NULL,
                    type TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    raw TEXT
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    txn_id TEXT NOT NULL,
                    xgb_score REAL NOT NULL,
                    gnn_score REAL,
                    blended_score REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    scored_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    latency_ms REAL NOT NULL,
                    FOREIGN KEY (txn_id) REFERENCES transactions(txn_id) ON DELETE CASCADE
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    txn_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'New',
                    assigned_to TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (txn_id) REFERENCES transactions(txn_id) ON DELETE CASCADE
                );
                """
            )
            # Recreate labels table if needed to ensure all columns match init.sql
            cur.execute("PRAGMA table_info(labels);")
            existing_label_cols = [r[1] for r in cur.fetchall()]
            if existing_label_cols and "disposition" not in existing_label_cols:
                cur.execute("DROP TABLE labels;")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    txn_id TEXT NOT NULL,
                    label INTEGER NOT NULL,
                    disposition TEXT NOT NULL,
                    previous_disposition TEXT,
                    analyst TEXT NOT NULL,
                    comment TEXT,
                    labeled_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (txn_id) REFERENCES transactions(txn_id) ON DELETE CASCADE
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_scores_txn_id ON scores (txn_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_labels_txn_id ON labels (txn_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_cases_status ON cases (status);"
            )
            conn.commit()

    @property
    def backend(self) -> str:
        """Return the active database backend ('postgresql' or 'sqlite')."""
        return self._backend

    def get_connection(self):
        """Get an active database connection (PostgreSQL or SQLite with Row factory)."""
        if self._backend == "postgresql":
            return psycopg2.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                dbname=settings.POSTGRES_DB,
            )
        else:
            db_path = Path(settings.SQLITE_DB_PATH)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_sqlite_schema(db_path)
            conn = sqlite3.connect(str(db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            return conn

    def is_reachable(self) -> bool:
        """Check if active database backend is reachable."""
        try:
            if self._backend == "postgresql":
                conn = psycopg2.connect(
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    dbname=settings.POSTGRES_DB,
                    connect_timeout=2,
                )
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                conn.close()
                return True
            elif self._backend == "sqlite":
                db_path = Path(settings.SQLITE_DB_PATH)
                if not db_path.exists():
                    return False
                with sqlite3.connect(str(db_path), timeout=2.0) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT 1;")
                return True
            return False
        except Exception as e:
            logger.warning("Database reachability check failed: %s", e)
            return False

    def save_transaction_and_score(
        self,
        txn_id: str,
        account_id: str,
        dest_id: str,
        amount: float,
        txn_type: str,
        ts: Optional[datetime],
        raw_payload: Dict[str, Any],
        xgb_score: float,
        gnn_score: Optional[float],
        blended_score: float,
        model_version: str,
        latency_ms: float,
    ) -> bool:
        """Enqueue transaction and score record for background persistence.

        Completely off the scoring path: returns in microseconds.
        """
        ts_iso = (
            ts.isoformat() if ts else datetime.now(timezone.utc).isoformat()
        )
        scored_at_iso = datetime.now(timezone.utc).isoformat()

        record = {
            "txn_id": txn_id,
            "account_id": account_id,
            "dest_id": dest_id,
            "amount": amount,
            "type": txn_type,
            "ts": ts_iso,
            "raw": raw_payload,
            "xgb_score": xgb_score,
            "gnn_score": gnn_score,
            "blended_score": blended_score,
            "model_version": model_version,
            "scored_at": scored_at_iso,
            "latency_ms": latency_ms,
        }

        try:
            self._write_queue.put_nowait(record)
            return True
        except queue.Full:
            with self._lock:
                self.failed_writes_count += 1
            logger.warning("Database write queue is full; dropping record for txn %s", txn_id)
            return False

    def flush(self, timeout: float = 5.0) -> None:
        """Wait for background queue to drain."""
        self._write_queue.join()

    def _process_write_queue(self) -> None:
        """Background worker thread draining write queue in batches."""
        batch_size = 500
        while self._running:
            items: List[Dict[str, Any]] = []
            try:
                # Block for first item
                item = self._write_queue.get(timeout=0.2)
                items.append(item)
                # Drain additional items up to batch_size without blocking
                while len(items) < batch_size:
                    try:
                        items.append(self._write_queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            if not items:
                continue

            # Persist batch
            try:
                if self._backend == "postgresql":
                    self._batch_insert_postgres(items)
                else:
                    self._batch_insert_sqlite(items)

                with self._lock:
                    self.successful_writes_count += len(items)
            except Exception as e:
                with self._lock:
                    self.failed_writes_count += len(items)
                logger.error("Failed to persist batch of %d records: %s", len(items), e)
            finally:
                for _ in items:
                    self._write_queue.task_done()

    def _batch_insert_sqlite(self, items: List[Dict[str, Any]]) -> None:
        """Batch insert into SQLite with a single transaction."""
        db_path = str(Path(settings.SQLITE_DB_PATH))
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            cur = conn.cursor()
            txn_tuples = [
                (
                    item["txn_id"],
                    item["account_id"],
                    item["dest_id"],
                    item["amount"],
                    item["type"],
                    item["ts"],
                    json.dumps(item["raw"]),
                )
                for item in items
            ]
            cur.executemany(
                """
                INSERT OR IGNORE INTO transactions (txn_id, account_id, dest_id, amount, type, ts, raw)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                txn_tuples,
            )

            score_tuples = [
                (
                    item["txn_id"],
                    item["xgb_score"],
                    item["gnn_score"],
                    item["blended_score"],
                    item["model_version"],
                    item["scored_at"],
                    item["latency_ms"],
                )
                for item in items
            ]
            cur.executemany(
                """
                INSERT INTO scores (
                    txn_id, xgb_score, gnn_score, blended_score, model_version, scored_at, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                score_tuples,
            )
            conn.commit()

    def _batch_insert_postgres(self, items: List[Dict[str, Any]]) -> None:
        """Batch insert into PostgreSQL."""
        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            dbname=settings.POSTGRES_DB,
            connect_timeout=5,
        )
        try:
            with conn.cursor() as cur:
                for item in items:
                    cur.execute(
                        """
                        INSERT INTO transactions (txn_id, account_id, dest_id, amount, type, ts, raw)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (txn_id) DO NOTHING;
                        """,
                        (
                            item["txn_id"],
                            item["account_id"],
                            item["dest_id"],
                            item["amount"],
                            item["type"],
                            item["ts"],
                            Json(item["raw"]),
                        ),
                    )
                    cur.execute(
                        """
                        INSERT INTO scores (
                            txn_id, xgb_score, gnn_score, blended_score, model_version, scored_at, latency_ms
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            item["txn_id"],
                            item["xgb_score"],
                            item["gnn_score"],
                            item["blended_score"],
                            item["model_version"],
                            item["scored_at"],
                            item["latency_ms"],
                        ),
                    )
            conn.commit()
        finally:
            conn.close()


# Global database manager instance
db_manager = DatabaseManager()
