"""
Persistence and Storage Layer.
Provides SQLite state registry, buffered Parquet ledger writer,
atomic 30s checkpoints (.tmp + os.replace), and SIGINT/SIGTERM graceful shutdown/resume.
"""

from __future__ import annotations

import json
import signal
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd


class StorageManager:
    """
    Manages SQLite database, Parquet ledgers, and atomic checkpoint files.
    """

    def __init__(self, run_id: str, base_dir: Path = Path("output/runs")):
        self.run_id = run_id
        self.run_dir = base_dir / run_id
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.leaderboard_dir = self.run_dir / "leaderboard"
        self.experiments_dir = self.run_dir / "experiments"
        self.images_dir = self.run_dir / "images"
        self.reports_dir = self.run_dir / "reports"

        for d in [
            self.run_dir,
            self.checkpoints_dir,
            self.leaderboard_dir,
            self.experiments_dir,
            self.images_dir,
            self.reports_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self.db_path = self.run_dir / "registry.sqlite"
        self.parquet_path = self.run_dir / "ledger.parquet"
        self.throughput_path = self.run_dir / "throughput.json"

        self._ledger_buffer: list[dict[str, Any]] = []
        self._last_flush_time = time.time()
        self._last_checkpoint_time = time.time()
        self._interrupted = False

        self._init_sqlite()
        self._setup_signal_handlers()

    def _init_sqlite(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    status TEXT,
                    b_star INTEGER,
                    config_json TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    exp_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    family TEXT,
                    genotype_hash TEXT,
                    signal_hash TEXT,
                    fitness REAL,
                    sharpe_dev REAL,
                    sharpe_basket REAL,
                    trades_primary INTEGER,
                    status TEXT,
                    created_at TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS funnel_stats (
                    run_id TEXT PRIMARY KEY,
                    n_generated INTEGER,
                    n_unique_genotype INTEGER,
                    n_unique_signal INTEGER,
                    n_s0 INTEGER,
                    n_s1 INTEGER,
                    n_s2 INTEGER,
                    n_s3 INTEGER,
                    updated_at TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _setup_signal_handlers(self) -> None:
        def handler(signum, frame):
            print(f"\n[STORAGE] Caught signal {signum} (SIGINT/SIGTERM). Saving atomic checkpoint before exit...")
            self._interrupted = True
            self.flush_ledger()

        # Handle signals if in main thread
        try:
            signal.signal(signal.SIGINT, handler)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, handler)
        except (ValueError, AttributeError):
            pass

    @property
    def is_interrupted(self) -> bool:
        return self._interrupted

    def record_run_start(self, config_dict: dict[str, Any], b_star: int | None = None) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, start_time, status, b_star, config_json) VALUES (?, ?, ?, ?, ?)",
                (self.run_id, pd.Timestamp.now("UTC").isoformat(), "RUNNING", b_star, json.dumps(config_dict)),
            )
            conn.commit()
        finally:
            conn.close()

    def update_run_b_star(self, b_star: int) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("UPDATE runs SET b_star = ? WHERE run_id = ?", (b_star, self.run_id))
            conn.commit()
        finally:
            conn.close()

    def record_run_end(self, candidates: list[dict[str, Any]] | None = None, holdout_meta: dict[str, Any] | None = None) -> None:
        """
        Marks run as COMPLETED in SQLite, updates leaderboard, and flushes ledger.
        """
        if candidates:
            self.update_leaderboard(candidates)
        self.flush_ledger()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "UPDATE runs SET status = ? WHERE run_id = ?",
                ("COMPLETED", self.run_id),
            )
            conn.commit()
        finally:
            conn.close()

    def log_evaluation(self, eval_data: dict[str, Any]) -> None:
        """
        Logs a strategy evaluation to buffer and SQLite.
        Flushes to Parquet every 30 seconds or if buffer exceeds 100 records.
        """
        self._ledger_buffer.append(eval_data)
        now = time.time()
        if now - self._last_flush_time >= 30.0 or len(self._ledger_buffer) >= 100:
            self.flush_ledger()

    def flush_ledger(self) -> None:
        if not self._ledger_buffer:
            return

        new_df = pd.DataFrame(self._ledger_buffer)
        if self.parquet_path.exists():
            existing_df = pd.read_parquet(self.parquet_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        # Atomic write
        tmp_path = self.parquet_path.with_suffix(".tmp")
        combined_df.to_parquet(tmp_path, index=False)
        tmp_path.replace(self.parquet_path)

        self._ledger_buffer.clear()
        self._last_flush_time = time.time()

    def save_checkpoint(
        self,
        checkpoint_data: dict[str, Any],
        step_name: str = "auto",
    ) -> Path:
        """
        Saves an atomic checkpoint file (.tmp + replace).
        """
        self.flush_ledger()
        checkpoint_file = self.checkpoints_dir / f"checkpoint_{step_name}.json"
        tmp_file = checkpoint_file.with_suffix(".tmp")

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        tmp_file.replace(checkpoint_file)
        # Also maintain latest
        latest_file = self.checkpoints_dir / "checkpoint_latest.json"
        latest_tmp = latest_file.with_suffix(".tmp")
        with open(latest_tmp, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        latest_tmp.replace(latest_file)

        self._last_checkpoint_time = time.time()
        return checkpoint_file

    def load_latest_checkpoint(self) -> dict[str, Any] | None:
        latest_file = self.checkpoints_dir / "checkpoint_latest.json"
        if not latest_file.exists():
            return None
        with open(latest_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_throughput_stats(self, stats: dict[str, int]) -> None:
        """
        Writes throughput.json atomically and syncs to SQLite.
        """
        tmp_file = self.throughput_path.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        tmp_file.replace(self.throughput_path)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO funnel_stats 
                (run_id, n_generated, n_unique_genotype, n_unique_signal, n_s0, n_s1, n_s2, n_s3, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    stats.get("N_generated", 0),
                    stats.get("N_unique_genotype", 0),
                    stats.get("N_unique_signal", 0),
                    stats.get("N_S0", 0),
                    stats.get("N_S1", 0),
                    stats.get("N_S2", 0),
                    stats.get("N_S3", 0),
                    pd.Timestamp.now("UTC").isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        """Flushes buffered ledgers and ensures clean state."""
        self.flush_ledger()

    def update_leaderboard(self, candidates: list[dict[str, Any]]) -> None:
        """
        Writes current leaderboard to current.json and current.csv.
        """
        json_path = self.leaderboard_dir / "current.json"
        csv_path = self.leaderboard_dir / "current.csv"

        tmp_json = json_path.with_suffix(".tmp")
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2, default=str)
        tmp_json.replace(json_path)

        if candidates:
            df = pd.DataFrame(candidates)
            tmp_csv = csv_path.with_suffix(".tmp")
            df.to_csv(tmp_csv, index=False)
            tmp_csv.replace(csv_path)
