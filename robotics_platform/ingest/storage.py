"""
SQLite local telemetry store with fast batch inserts, indexing, and Parquet export.
"""

from typing import List, Dict, Any, Optional
import os
import json
import sqlite3
import pandas as pd

from robotics_platform.ingest.schema import TelemetryEvent


class TelemetryStorage:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry_events (
                event_id TEXT PRIMARY KEY,
                robot_id TEXT NOT NULL,
                stream_type TEXT NOT NULL,
                event_timestamp REAL NOT NULL,
                ingest_timestamp REAL NOT NULL,
                corrected_timestamp REAL,
                sequence_number INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                drop_reason TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_robot_stream 
            ON telemetry_events(robot_id, stream_type, ingest_timestamp)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_status
            ON telemetry_events(status)
            """
        )
        self.conn.commit()

    def insert_batch(self, events: List[TelemetryEvent]):
        if not events:
            return
        records = [
            (
                e.event_id,
                e.robot_id,
                e.stream_type,
                e.event_timestamp,
                e.ingest_timestamp,
                e.corrected_timestamp,
                e.sequence_number,
                json.dumps(e.payload),
                e.status,
                e.drop_reason,
            )
            for e in events
        ]
        cursor = self.conn.cursor()
        cursor.executemany(
            """
            INSERT OR REPLACE INTO telemetry_events (
                event_id, robot_id, stream_type, event_timestamp,
                ingest_timestamp, corrected_timestamp, sequence_number,
                payload_json, status, drop_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        self.conn.commit()

    def count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM telemetry_events")
        return cursor.fetchone()[0]

    def query(
        self,
        robot_id: Optional[str] = None,
        stream_type: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        query = "SELECT * FROM telemetry_events WHERE 1=1"
        params = []
        if robot_id:
            query += " AND robot_id = ?"
            params.append(robot_id)
        if stream_type:
            query += " AND stream_type = ?"
            params.append(stream_type)
        query += " ORDER BY ingest_timestamp ASC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, r)) for r in rows]

    def export_parquet(self, filepath: str):
        """Exports all telemetry events to Apache Parquet for downstream analytics."""
        df = pd.read_sql_query("SELECT * FROM telemetry_events", self.conn)
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        df.to_parquet(filepath, index=False)
        return len(df)

    def close(self):
        self.conn.close()
