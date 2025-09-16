from __future__ import annotations
from typing import List, Optional, Tuple
import sqlite3
from datetime import datetime


class UserRepository:
    def __init__(self, conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
        self.conn = conn
        self.c = cursor

    def get_password(self, username: str) -> Optional[str]:
        self.c.execute("SELECT password FROM users WHERE username=?", (username,))
        row = self.c.fetchone()
        return row[0] if row else None

    def user_exists(self, username: str) -> bool:
        self.c.execute("SELECT 1 FROM users WHERE username=?", (username,))
        return self.c.fetchone() is not None

    def create_user(self, username: str, password_hash: str) -> None:
        self.c.execute("INSERT INTO users VALUES (?, ?)", (username, password_hash))
        self.conn.commit()


class HistoryRepository:
    def __init__(self, conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
        self.conn = conn
        self.c = cursor

    def list_by_user(self, user: str) -> List[Tuple[int, str]]:
        self.c.execute("SELECT convo_id, title FROM history WHERE user=?", (user,))
        return self.c.fetchall()

    def insert_history(self, user: str, title: str, messages_json: str) -> int:
        self.c.execute("INSERT INTO history (user, title, messages) VALUES (?, ?, ?)", (user, title, messages_json))
        self.conn.commit()
        return self.c.lastrowid

    def update_history(self, convo_id: int, title: str, messages_json: str) -> None:
        self.c.execute("UPDATE history SET title=?, messages=? WHERE convo_id=?", (title, messages_json, convo_id))
        self.conn.commit()

    def get_messages(self, convo_id: int) -> Optional[str]:
        self.c.execute("SELECT messages FROM history WHERE convo_id=?", (convo_id,))
        row = self.c.fetchone()
        return row[0] if row else None

    def delete_history(self, convo_id: int) -> None:
        self.c.execute("DELETE FROM history WHERE convo_id=?", (convo_id,))
        self.conn.commit()


class MemoryRepository:
    def __init__(self, conn: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
        self.conn = conn
        self.c = cursor

    def upsert_value(self, user: str, convo_id: int, mem_key: str, mem_value_json: str) -> None:
        self.c.execute(
            "INSERT OR REPLACE INTO memory (user, convo_id, mem_key, mem_value) VALUES (?, ?, ?, ?)",
            (user, convo_id, mem_key, mem_value_json),
        )
        # no commit here; caller decides

    def get_value(self, user: str, convo_id: int, mem_key: str) -> Optional[str]:
        self.c.execute(
            "SELECT mem_value FROM memory WHERE user=? AND convo_id=? AND mem_key=? ORDER BY timestamp DESC LIMIT 1",
            (user, convo_id, mem_key),
        )
        row = self.c.fetchone()
        return row[0] if row else None

    def get_recent(self, user: str, convo_id: int, limit: int) -> List[Tuple[str, str]]:
        self.c.execute(
            "SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=? ORDER BY timestamp DESC LIMIT ?",
            (user, convo_id, limit),
        )
        return self.c.fetchall()

    def insert_semantic(self, user: str, convo_id: int, mem_key: str, json_semantic: str, salience: float, ts: datetime) -> int:
        self.c.execute(
            "INSERT OR REPLACE INTO memory (user, convo_id, mem_key, mem_value, salience, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (user, convo_id, mem_key, json_semantic, salience, ts),
        )
        return self.c.lastrowid

    def insert_episodic(self, user: str, convo_id: int, mem_key: str, json_episodic: str, salience: float, parent_id: Optional[int], ts: datetime) -> None:
        self.c.execute(
            "INSERT OR REPLACE INTO memory (user, convo_id, mem_key, mem_value, salience, parent_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user, convo_id, mem_key, json_episodic, salience, parent_id, ts),
        )

    def get_recent_with_salience(self, user: str, convo_id: int, limit: int) -> List[Tuple[str, str, float]]:
        self.c.execute(
            "SELECT mem_key, mem_value, salience FROM memory WHERE user=? AND convo_id=? ORDER BY timestamp DESC LIMIT ?",
            (user, convo_id, limit),
        )
        return self.c.fetchall()

    def decay_salience(self, user: str, convo_id: int, decay_factor: float, cutoff_dt: datetime) -> None:
        self.c.execute(
            "UPDATE memory SET salience = salience * ? WHERE user=? AND convo_id=? AND timestamp < ?",
            (decay_factor, user, convo_id, cutoff_dt),
        )

    def delete_low_salience(self, user: str, convo_id: int, threshold: float) -> None:
        self.c.execute(
            "DELETE FROM memory WHERE user=? AND convo_id=? AND salience < ?",
            (user, convo_id, threshold),
        )