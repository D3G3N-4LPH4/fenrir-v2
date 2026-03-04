#!/usr/bin/env python3
"""
FENRIR - Merkle Hash-Chain Audit Trail

Cryptographically linked, tamper-evident log of every action the bot takes.
Each record includes SHA256(prev_hash + event_type + payload + timestamp),
creating an unbreakable chain. Modify one entry and the entire chain
downstream becomes invalid.

Inspired by OpenFang's Merkle audit system, adapted for trading operations.

Usage:
    audit = AuditChain(db_path="fenrir_trades.db")

    # Record events throughout the bot lifecycle
    audit.record("TOKEN_DETECTED", "ABC123", {"symbol": "WOLF", "liquidity": 5.0})
    audit.record("AI_EVAL_ENTRY", "ABC123", {"decision": "BUY", "confidence": 0.85})
    audit.record("BUY_EXECUTED", "ABC123", {"amount_sol": 0.1, "signature": "sig..."})

    # Verify chain integrity
    valid, broken_at = audit.verify_chain()

    # Replay a session
    events = audit.get_session_log(session_id="abc-123")
"""

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
#                           EVENT TYPES
# ═══════════════════════════════════════════════════════════════════════════

class AuditEventType:
    """Constants for audit event types."""

    TOKEN_DETECTED = "TOKEN_DETECTED"  # noqa: S105
    TOKEN_FILTERED = "TOKEN_FILTERED"  # noqa: S105
    AI_EVAL_ENTRY = "AI_EVAL_ENTRY"
    AI_EVAL_EXIT = "AI_EVAL_EXIT"
    AI_OVERRIDE = "AI_OVERRIDE"
    AI_TIMEOUT = "AI_TIMEOUT"
    AI_ERROR = "AI_ERROR"
    BUY_EXECUTED = "BUY_EXECUTED"
    BUY_FAILED = "BUY_FAILED"
    SELL_EXECUTED = "SELL_EXECUTED"
    SELL_FAILED = "SELL_FAILED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    PRICE_UPDATE = "PRICE_UPDATE"
    MECHANICAL_TRIGGER = "MECHANICAL_TRIGGER"
    STRATEGY_ACTIVATED = "STRATEGY_ACTIVATED"
    STRATEGY_PAUSED = "STRATEGY_PAUSED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    BOT_STARTED = "BOT_STARTED"
    BOT_STOPPED = "BOT_STOPPED"
    ERROR = "ERROR"


GENESIS_SEED = "FENRIR_GENESIS_v2"


# ═══════════════════════════════════════════════════════════════════════════
#                           AUDIT RECORD
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AuditRecord:
    """A single entry in the audit chain."""

    id: int | None = None
    session_id: str = ""
    timestamp: str = ""
    event_type: str = ""
    token_address: str | None = None
    strategy_id: str | None = None
    payload: dict = field(default_factory=dict)
    prev_hash: str = ""
    hash: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "token_address": self.token_address,
            "strategy_id": self.strategy_id,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }


# ═══════════════════════════════════════════════════════════════════════════
#                           AUDIT CHAIN
# ═══════════════════════════════════════════════════════════════════════════

class AuditChain:
    """
    Tamper-evident audit trail using SHA256 hash chaining.

    Every record's hash includes the previous record's hash, creating
    a Merkle-like chain. Break one link and verification fails for
    everything downstream.
    """

    def __init__(self, db_path: str = "fenrir_trades.db", session_id: str | None = None):
        self.db_path = db_path
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.conn: sqlite3.Connection | None = None
        self._last_hash: str = ""
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Create audit table and load the last hash for chain continuity."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_chain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                token_address TEXT,
                strategy_id TEXT,
                payload TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL
            )
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_session
            ON audit_chain (session_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp
            ON audit_chain (timestamp)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_token
            ON audit_chain (token_address)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_event_type
            ON audit_chain (event_type)
        """)

        self.conn.commit()

        # Load the last hash to continue the chain
        row = self.conn.execute(
            "SELECT hash FROM audit_chain ORDER BY id DESC LIMIT 1"
        ).fetchone()

        if row:
            self._last_hash = row["hash"]
        else:
            # Genesis: hash of the seed phrase
            self._last_hash = hashlib.sha256(GENESIS_SEED.encode()).hexdigest()

    @staticmethod
    def _compute_hash(
        prev_hash: str,
        event_type: str,
        payload_json: str,
        timestamp: str,
    ) -> str:
        """Compute SHA256 hash for a chain entry."""
        preimage = f"{prev_hash}|{event_type}|{payload_json}|{timestamp}"
        return hashlib.sha256(preimage.encode("utf-8")).hexdigest()

    def record(
        self,
        event_type: str,
        token_address: str | None = None,
        payload: dict | None = None,
        strategy_id: str | None = None,
    ) -> AuditRecord:
        """
        Record an event in the audit chain.

        Args:
            event_type: One of AuditEventType constants
            token_address: Token mint address (if applicable)
            payload: Event-specific data dict
            strategy_id: Which strategy triggered this event

        Returns:
            The recorded AuditRecord with computed hash
        """
        payload = payload or {}
        timestamp = datetime.now().isoformat()
        payload_json = json.dumps(payload, sort_keys=True, default=str)

        # Compute chained hash
        new_hash = self._compute_hash(
            self._last_hash, event_type, payload_json, timestamp
        )

        cursor = self.conn.execute(
            """
            INSERT INTO audit_chain
                (session_id, timestamp, event_type, token_address,
                 strategy_id, payload, prev_hash, hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.session_id,
                timestamp,
                event_type,
                token_address,
                strategy_id,
                payload_json,
                self._last_hash,
                new_hash,
            ),
        )
        self.conn.commit()

        record = AuditRecord(
            id=cursor.lastrowid,
            session_id=self.session_id,
            timestamp=timestamp,
            event_type=event_type,
            token_address=token_address,
            strategy_id=strategy_id,
            payload=payload,
            prev_hash=self._last_hash,
            hash=new_hash,
        )

        self._last_hash = new_hash
        return record

    def verify_chain(
        self,
        session_id: str | None = None,
    ) -> tuple[bool, int | None]:
        """
        Verify the integrity of the audit chain.

        Args:
            session_id: If provided, verify only that session's entries.
                        If None, verify the entire chain.

        Returns:
            (is_valid, broken_at_id)
            - is_valid: True if entire chain is intact
            - broken_at_id: ID of first broken record, or None if valid
        """
        if session_id:
            rows = self.conn.execute(
                "SELECT * FROM audit_chain WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM audit_chain ORDER BY id ASC"
            ).fetchall()

        if not rows:
            return (True, None)

        for row in rows:
            expected_hash = self._compute_hash(
                row["prev_hash"],
                row["event_type"],
                row["payload"],
                row["timestamp"],
            )

            if expected_hash != row["hash"]:
                return (False, row["id"])

        # Also verify chain linkage (each prev_hash matches prior row's hash)
        for i in range(1, len(rows)):
            if rows[i]["prev_hash"] != rows[i - 1]["hash"]:
                return (False, rows[i]["id"])

        return (True, None)

    def get_session_log(
        self,
        session_id: str | None = None,
        event_type: str | None = None,
        token_address: str | None = None,
        limit: int = 500,
    ) -> list[AuditRecord]:
        """
        Retrieve audit records with optional filters.

        Args:
            session_id: Filter by session (defaults to current)
            event_type: Filter by event type
            token_address: Filter by token
            limit: Max records to return
        """
        sid = session_id or self.session_id
        conditions = ["session_id = ?"]
        params: list = [sid]

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)

        if token_address:
            conditions.append("token_address = ?")
            params.append(token_address)

        where_clause = " AND ".join(conditions)
        params.append(limit)

        rows = self.conn.execute(
            f"SELECT * FROM audit_chain WHERE {where_clause} ORDER BY id ASC LIMIT ?",  # noqa: S608
            params,
        ).fetchall()

        records = []
        for row in rows:
            records.append(
                AuditRecord(
                    id=row["id"],
                    session_id=row["session_id"],
                    timestamp=row["timestamp"],
                    event_type=row["event_type"],
                    token_address=row["token_address"],
                    strategy_id=row["strategy_id"],
                    payload=json.loads(row["payload"]),
                    prev_hash=row["prev_hash"],
                    hash=row["hash"],
                )
            )
        return records

    def get_token_timeline(self, token_address: str) -> list[AuditRecord]:
        """Get the complete audit trail for a specific token across all sessions."""
        rows = self.conn.execute(
            "SELECT * FROM audit_chain WHERE token_address = ? ORDER BY id ASC",
            (token_address,),
        ).fetchall()

        return [
            AuditRecord(
                id=row["id"],
                session_id=row["session_id"],
                timestamp=row["timestamp"],
                event_type=row["event_type"],
                token_address=row["token_address"],
                strategy_id=row["strategy_id"],
                payload=json.loads(row["payload"]),
                prev_hash=row["prev_hash"],
                hash=row["hash"],
            )
            for row in rows
        ]

    def get_chain_stats(self) -> dict:
        """Summary statistics for the audit chain."""
        row = self.conn.execute(
            "SELECT COUNT(*) as total, MIN(timestamp) as first, MAX(timestamp) as last "
            "FROM audit_chain"
        ).fetchone()

        sessions = self.conn.execute(
            "SELECT COUNT(DISTINCT session_id) as count FROM audit_chain"
        ).fetchone()

        return {
            "total_records": row["total"],
            "first_record": row["first"],
            "last_record": row["last"],
            "total_sessions": sessions["count"],
            "current_session": self.session_id,
            "last_hash": self._last_hash,
        }

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
