#!/usr/bin/env python3
"""
Tests for AISessionMemory.from_audit_chain (§1 harness kernel).

Projects the append-only audit chain into session memory and verifies the
reconstruction is faithful to the live record_decision → buy → update_outcome
path: decisions, was_bought, outcomes, rolling tallies, risk-appetite signal,
and rendered context block.

Payloads here mirror exactly what AuditAdapter writes (event.data + symbol).
"""

from fenrir.ai.memory import AISessionMemory
from fenrir.data.audit import AuditChain


# ═══════════════════════════════════════════════════════════════════
#  Fixtures / helpers
# ═══════════════════════════════════════════════════════════════════

def make_chain(tmp_path) -> AuditChain:
    return AuditChain(db_path=str(tmp_path / "audit.db"))


def ai_decision(chain, addr, symbol, decision, confidence, risk, reasoning="r"):
    chain.record(
        "AI_DECISION",
        token_address=addr,
        payload={
            "symbol": symbol,
            "decision": decision,
            "confidence": confidence,
            "risk_score": risk,
            "reasoning": reasoning,
        },
    )


def buy(chain, addr, symbol):
    chain.record(
        "BUY_EXECUTED",
        token_address=addr,
        payload={"symbol": symbol, "amount_sol": 0.1, "entry_price": 0.001},
    )


def sell(chain, addr, symbol, pnl_pct, pnl_sol, reason="exit", hold=5):
    chain.record(
        "SELL_EXECUTED",
        token_address=addr,
        payload={
            "symbol": symbol,
            "pnl_pct": pnl_pct,
            "pnl_sol": pnl_sol,
            "reason": reason,
            "hold_minutes": hold,
        },
    )


# ═══════════════════════════════════════════════════════════════════
#  Basic projection
# ═══════════════════════════════════════════════════════════════════

def test_empty_chain_yields_empty_memory(tmp_path):
    chain = make_chain(tmp_path)
    mem = AISessionMemory.from_audit_chain(chain)
    assert mem.decisions == []
    assert mem.get_session_stats()["total_decisions"] == 0


def test_full_lifecycle_projection(tmp_path):
    chain = make_chain(tmp_path)
    ai_decision(chain, "WOLF", "WOLF", "BUY", 0.8, 4.0, "strong setup")
    buy(chain, "WOLF", "WOLF")
    sell(chain, "WOLF", "WOLF", pnl_pct=42.0, pnl_sol=0.042, reason="take profit", hold=8)

    mem = AISessionMemory.from_audit_chain(chain)

    assert len(mem.decisions) == 1
    d = mem.decisions[0]
    assert d.token_mint == "WOLF"
    assert d.decision == "BUY"
    assert d.confidence == 0.8
    assert d.was_bought is True
    assert d.outcome_pnl_pct == 42.0
    assert d.outcome_hold_time_minutes == 8

    stats = mem.get_session_stats()
    assert stats["total_buys"] == 1
    assert stats["total_skips"] == 0
    assert stats["closed_trades"] == 1
    assert stats["profitable_trades"] == 1
    assert stats["total_pnl_sol"] == 0.042


def test_skip_decision_not_marked_bought(tmp_path):
    chain = make_chain(tmp_path)
    ai_decision(chain, "RUG", "RUG", "SKIP", 0.2, 9.0, "too concentrated")

    mem = AISessionMemory.from_audit_chain(chain)

    assert mem.decisions[0].was_bought is False
    assert mem.get_session_stats()["total_skips"] == 1
    assert mem.get_session_stats()["total_buys"] == 0


def test_auto_buy_without_ai_decision_is_synthesized(tmp_path):
    # AI disabled path: a BUY/SELL with no preceding AI_DECISION still surfaces.
    chain = make_chain(tmp_path)
    buy(chain, "AUTO", "AUTO")
    sell(chain, "AUTO", "AUTO", pnl_pct=-30.0, pnl_sol=-0.03)

    mem = AISessionMemory.from_audit_chain(chain)

    assert len(mem.decisions) == 1
    assert mem.decisions[0].was_bought is True
    assert mem.decisions[0].outcome_pnl_pct == -30.0
    assert mem.get_session_stats()["closed_trades"] == 1
    assert mem.get_session_stats()["profitable_trades"] == 0


# ═══════════════════════════════════════════════════════════════════
#  Rolling buffer + tallies
# ═══════════════════════════════════════════════════════════════════

def test_max_size_evicts_but_tallies_count_all(tmp_path):
    chain = make_chain(tmp_path)
    for i in range(10):
        ai_decision(chain, f"T{i}", f"T{i}", "BUY", 0.7, 5.0)
        buy(chain, f"T{i}", f"T{i}")

    mem = AISessionMemory.from_audit_chain(chain, max_size=3)

    # Deque holds only the last 3 …
    assert len(mem.decisions) == 3
    assert [d.token_mint for d in mem.decisions] == ["T7", "T8", "T9"]
    # … but tallies reflect every decision ever recorded.
    assert mem.get_session_stats()["total_buys"] == 10


# ═══════════════════════════════════════════════════════════════════
#  Risk-appetite signal survives the round trip
# ═══════════════════════════════════════════════════════════════════

def test_losing_streak_reconstructed(tmp_path):
    chain = make_chain(tmp_path)
    for i in range(3):
        addr = f"L{i}"
        ai_decision(chain, addr, addr, "BUY", 0.7, 5.0)
        buy(chain, addr, addr)
        sell(chain, addr, addr, pnl_pct=-20.0, pnl_sol=-0.02)

    mem = AISessionMemory.from_audit_chain(chain)

    assert mem.get_session_stats()["closed_trades"] == 3
    assert mem.get_session_stats()["profitable_trades"] == 0
    assert "consecutive losing trades" in mem.get_risk_appetite_adjustment()


# ═══════════════════════════════════════════════════════════════════
#  Rendered context block
# ═══════════════════════════════════════════════════════════════════

def test_context_block_rendered_from_projection(tmp_path):
    chain = make_chain(tmp_path)
    ai_decision(chain, "MOON", "MOON", "STRONG_BUY", 0.9, 3.0, "clean chart")
    buy(chain, "MOON", "MOON")
    sell(chain, "MOON", "MOON", pnl_pct=120.0, pnl_sol=0.12)

    mem = AISessionMemory.from_audit_chain(chain)
    block = mem.build_context_block()

    assert "STRONG_BUY $MOON" in block
    assert "RESULT: +120.0%" in block
    assert "SESSION PERFORMANCE" in block


# ═══════════════════════════════════════════════════════════════════
#  Equivalence with the live path
# ═══════════════════════════════════════════════════════════════════

def test_projection_matches_live_tallies(tmp_path):
    """Projecting the log should produce the same tallies as driving the live
    record_decision/update_outcome methods with the equivalent sequence."""
    from datetime import datetime

    from fenrir.ai.memory import DecisionRecord

    chain = make_chain(tmp_path)
    live = AISessionMemory(max_size=15)

    seq = [
        ("A", "BUY", 0.8, 4.0, True, 50.0, 0.05),
        ("B", "SKIP", 0.3, 8.0, False, None, None),
        ("C", "BUY", 0.7, 5.0, True, -25.0, -0.025),
    ]
    for addr, decision, conf, risk, bought, pnl, pnl_sol in seq:
        ai_decision(chain, addr, addr, decision, conf, risk)
        rec = DecisionRecord(
            timestamp=datetime.now(), token_mint=addr, token_symbol=addr,
            token_name=addr, decision=decision, confidence=conf,
            risk_score=risk, reasoning_summary="r", was_bought=bought,
        )
        live.record_decision(rec)
        if bought:
            # Bought rows always carry concrete pnl values; narrow for the type checker.
            assert pnl is not None and pnl_sol is not None
            buy(chain, addr, addr)
            sell(chain, addr, addr, pnl_pct=pnl, pnl_sol=pnl_sol)
            live.update_outcome(addr, pnl, "exit", 5, pnl_sol)

    projected = AISessionMemory.from_audit_chain(chain)

    ls, ps = live.get_session_stats(), projected.get_session_stats()
    for key in ("total_buys", "total_skips", "closed_trades", "profitable_trades"):
        assert ls[key] == ps[key], key
    assert ls["total_pnl_sol"] == projected.get_session_stats()["total_pnl_sol"]
