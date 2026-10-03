"""Landing book. The trading system is the slot gap, not the strategy name.

Log create-slot versus confirmation-slot for every ignition attempt.
Promote a lane only after a week of these rows stays positive after tips.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class SlotBook:
    def __init__(self, path: str = "data/slot_book.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        mint: str,
        create_slot: int | None,
        landed_slot: int | None,
        filled: bool,
        amount_sol: float,
        tip_lamports: int,
        pnl_sol: float | None = None,
    ) -> dict:
        offset = None
        if create_slot is not None and landed_slot is not None:
            offset = landed_slot - create_slot
        row = {
            "ts": time.time(),
            "mint": mint,
            "create_slot": create_slot,
            "landed_slot": landed_slot,
            "slot_offset": offset,
            "filled": filled,
            "amount_sol": amount_sol,
            "tip_lamports": tip_lamports,
            "pnl_sol": pnl_sol,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        return row

    def summary(self) -> dict:
        if not self.path.exists():
            return {
                "attempts": 0,
                "fill_rate": 0.0,
                "median_slot_offset": None,
                "expectancy_sol": None,
            }
        rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        offsets = [r["slot_offset"] for r in rows if r["slot_offset"] is not None]
        fills = [r for r in rows if r["filled"]]
        pnl = [r["pnl_sol"] for r in rows if r["pnl_sol"] is not None]
        return {
            "attempts": len(rows),
            "fill_rate": (len(fills) / len(rows)) if rows else 0.0,
            "median_slot_offset": sorted(offsets)[len(offsets) // 2] if offsets else None,
            "expectancy_sol": (sum(pnl) / len(pnl)) if pnl else None,
        }
