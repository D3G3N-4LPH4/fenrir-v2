#!/usr/bin/env python3
"""
FENRIR - pump.fun Instruction Test Suite

Pins the rebuilt (current-IDL) buy/sell instruction layout: account count,
order, writability, discriminators, args, PDA/ATA derivations, Token-2022
support, and creator decoding. Verified against real on-chain txs; these tests
keep it from silently drifting again.

Run with: pytest tests/test_pumpfun_instructions.py -v
"""

from __future__ import annotations

import struct

from solders.pubkey import Pubkey

from fenrir.protocol.pumpfun import (
    ASSOCIATED_TOKEN_PROGRAM,
    BUY_DISCRIMINATOR,
    PUMP_BUYBACK_FEE_RECIPIENT,
    PUMP_EVENT_AUTHORITY,
    PUMP_FEE_CONFIG,
    PUMP_FEE_POOL_RECIPIENT,
    PUMP_FEE_PROGRAM,
    PUMP_FEE_RECIPIENT,
    PUMP_GLOBAL,
    PUMP_PROGRAM_ID,
    SELL_DISCRIMINATOR,
    TOKEN_2022_PROGRAM,
    TOKEN_PROGRAM,
    BondingCurveState,
    PumpFunProgram,
)

PF = PumpFunProgram()
BUYER = Pubkey.from_string("So11111111111111111111111111111111111111112")
MINT = Pubkey.from_string("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
CREATOR = Pubkey.from_string("Vote111111111111111111111111111111111111111")
BC, _ = PF.derive_bonding_curve_address(MINT)


def _pk(meta) -> str:
    return str(meta.pubkey)


class TestBuyInstruction:
    def _ix(self, token_program: Pubkey = TOKEN_PROGRAM):
        return PF.build_buy_instruction(
            buyer=BUYER,
            token_mint=MINT,
            bonding_curve=BC,
            creator=CREATOR,
            token_program=token_program,
            amount_tokens=1_000_000,
            max_sol_cost=1_050_000,
        )

    def test_program_and_count(self) -> None:
        ix = self._ix()
        assert ix.program_id == PUMP_PROGRAM_ID
        assert len(ix.accounts) == 18  # 16 IDL + 2 v2 buyback fee accounts

    def test_data(self) -> None:
        ix = self._ix()
        data = bytes(ix.data)
        assert data[:8] == BUY_DISCRIMINATOR
        assert struct.unpack("<Q", data[8:16])[0] == 1_000_000  # amount (token base units)
        assert struct.unpack("<Q", data[16:24])[0] == 1_050_000  # max_sol_cost (lamports)
        assert data[24:] == b"\x00"  # track_volume = None
        assert len(data) == 25

    def test_account_order(self) -> None:
        a = self._ix().accounts
        assert _pk(a[0]) == str(PUMP_GLOBAL)
        assert _pk(a[1]) == str(PUMP_FEE_RECIPIENT) and a[1].is_writable
        assert _pk(a[2]) == str(MINT)
        assert _pk(a[3]) == str(BC) and a[3].is_writable
        assert _pk(a[4]) == str(PF.derive_ata(BC, MINT, TOKEN_PROGRAM))
        assert _pk(a[5]) == str(PF.derive_ata(BUYER, MINT, TOKEN_PROGRAM))
        assert _pk(a[6]) == str(BUYER) and a[6].is_signer and a[6].is_writable
        assert _pk(a[8]) == str(TOKEN_PROGRAM)
        assert _pk(a[9]) == str(PF.derive_creator_vault(CREATOR)) and a[9].is_writable
        assert _pk(a[10]) == str(PUMP_EVENT_AUTHORITY)
        assert _pk(a[11]) == str(PUMP_PROGRAM_ID)
        assert _pk(a[12]) == str(PF.derive_global_volume_accumulator())
        assert _pk(a[13]) == str(PF.derive_user_volume_accumulator(BUYER)) and a[13].is_writable
        assert _pk(a[14]) == str(PUMP_FEE_CONFIG)
        assert _pk(a[15]) == str(PUMP_FEE_PROGRAM)
        # v2 buyback fee accounts — appended, both writable, order-sensitive.
        assert _pk(a[16]) == str(PUMP_BUYBACK_FEE_RECIPIENT) and a[16].is_writable
        assert _pk(a[17]) == str(PUMP_FEE_POOL_RECIPIENT) and a[17].is_writable

    def test_fee_recipient_override(self) -> None:
        override = Pubkey.from_string("62qc2CNXwrYqQScmEdiZFFAnJR262PxWEuNQtxfafNgV")
        ix = PF.build_buy_instruction(
            buyer=BUYER,
            token_mint=MINT,
            bonding_curve=BC,
            creator=CREATOR,
            token_program=TOKEN_PROGRAM,
            amount_tokens=1,
            max_sol_cost=1,
            fee_recipient=override,
        )
        assert _pk(ix.accounts[1]) == str(override)

    def test_token_2022_changes_atas(self) -> None:
        classic = self._ix(TOKEN_PROGRAM).accounts
        t22 = self._ix(TOKEN_2022_PROGRAM).accounts
        assert _pk(t22[8]) == str(TOKEN_2022_PROGRAM)
        assert _pk(t22[4]) != _pk(classic[4])  # associated_bonding_curve differs
        assert _pk(t22[5]) != _pk(classic[5])  # associated_user differs
        assert _pk(t22[4]) == str(PF.derive_ata(BC, MINT, TOKEN_2022_PROGRAM))


class TestSellInstruction:
    def _ix(self, token_program: Pubkey = TOKEN_PROGRAM):
        return PF.build_sell_instruction(
            seller=BUYER,
            token_mint=MINT,
            bonding_curve=BC,
            creator=CREATOR,
            token_program=token_program,
            amount_tokens=5_000,
            min_sol_output=10,
        )

    def test_program_and_count(self) -> None:
        ix = self._ix()
        assert ix.program_id == PUMP_PROGRAM_ID
        assert len(ix.accounts) == 16  # 14 IDL + 2 v2 buyback fee accounts

    def test_data(self) -> None:
        data = bytes(self._ix().data)
        assert data[:8] == SELL_DISCRIMINATOR
        assert struct.unpack("<Q", data[8:16])[0] == 5_000
        assert struct.unpack("<Q", data[16:24])[0] == 10
        assert len(data) == 24  # no track_volume on sell

    def test_creator_vault_before_token_program(self) -> None:
        a = self._ix().accounts
        # The key buy/sell difference: sell has creator_vault at 8, token at 9.
        assert _pk(a[8]) == str(PF.derive_creator_vault(CREATOR)) and a[8].is_writable
        assert _pk(a[9]) == str(TOKEN_PROGRAM)
        assert _pk(a[10]) == str(PUMP_EVENT_AUTHORITY)
        assert _pk(a[11]) == str(PUMP_PROGRAM_ID)
        assert _pk(a[12]) == str(PUMP_FEE_CONFIG)
        assert _pk(a[13]) == str(PUMP_FEE_PROGRAM)
        # v2 buyback fee accounts — appended, both writable.
        assert _pk(a[14]) == str(PUMP_BUYBACK_FEE_RECIPIENT) and a[14].is_writable
        assert _pk(a[15]) == str(PUMP_FEE_POOL_RECIPIENT) and a[15].is_writable

    def test_extra_accounts_replace_legacy_tail(self) -> None:
        # When a shadowed cashback tail is supplied, it replaces the legacy
        # two-account tail verbatim (base 14 IDL accounts kept unchanged).
        from solders.instruction import AccountMeta

        uva = PF.derive_user_volume_accumulator(BUYER)
        bcv2 = Pubkey.find_program_address([b"bonding-curve-v2", bytes(MINT)], PUMP_PROGRAM_ID)[0]
        fee_rot = Pubkey.from_string("11111111111111111111111111111112")
        tail = [
            AccountMeta(pubkey=uva, is_signer=False, is_writable=True),
            AccountMeta(pubkey=bcv2, is_signer=False, is_writable=False),
            AccountMeta(pubkey=fee_rot, is_signer=False, is_writable=True),
        ]
        ix = PF.build_sell_instruction(
            seller=BUYER,
            token_mint=MINT,
            bonding_curve=BC,
            creator=CREATOR,
            token_program=TOKEN_PROGRAM,
            amount_tokens=5_000,
            min_sol_output=10,
            extra_accounts=tail,
        )
        a = ix.accounts
        assert len(a) == 17  # 14 IDL + 3 shadowed cashback tail
        assert _pk(a[13]) == str(PUMP_FEE_PROGRAM)  # base unchanged
        assert _pk(a[14]) == str(uva) and a[14].is_writable
        assert _pk(a[15]) == str(bcv2) and not a[15].is_writable
        assert _pk(a[16]) == str(fee_rot) and a[16].is_writable


class TestCreateAta:
    def test_idempotent_create_ata(self) -> None:
        ix = PF.build_create_ata_instruction(BUYER, BUYER, MINT, TOKEN_PROGRAM)
        assert ix.program_id == ASSOCIATED_TOKEN_PROGRAM
        assert bytes(ix.data) == bytes([1])  # CreateIdempotent
        assert len(ix.accounts) == 6
        assert _pk(ix.accounts[1]) == str(PF.derive_ata(BUYER, MINT, TOKEN_PROGRAM))
        assert _pk(ix.accounts[5]) == str(TOKEN_PROGRAM)


class TestParseGlobalFeeRecipient:
    def test_reads_offset_41(self) -> None:
        want = Pubkey.from_string("62qc2CNXwrYqQScmEdiZFFAnJR262PxWEuNQtxfafNgV")
        data = b"\x00" * 41 + bytes(want) + b"\x00" * 100
        assert PF.parse_global_fee_recipient(data) == want

    def test_none_when_short(self) -> None:
        assert PF.parse_global_fee_recipient(b"\x00" * 40) is None


class TestDeriveHelpers:
    def test_pda_seeds(self) -> None:
        assert (
            PF.derive_creator_vault(CREATOR)
            == Pubkey.find_program_address([b"creator-vault", bytes(CREATOR)], PUMP_PROGRAM_ID)[0]
        )
        assert (
            PF.derive_global_volume_accumulator()
            == Pubkey.find_program_address([b"global_volume_accumulator"], PUMP_PROGRAM_ID)[0]
        )
        assert (
            PF.derive_user_volume_accumulator(BUYER)
            == Pubkey.find_program_address(
                [b"user_volume_accumulator", bytes(BUYER)], PUMP_PROGRAM_ID
            )[0]
        )


class TestDecodeCreator:
    def _curve(self, creator: Pubkey | None) -> bytes:
        d = b"\x00" * 8  # discriminator
        d += struct.pack("<Q", 1_073_000_000)
        d += struct.pack("<Q", 30_000_000_000)
        d += struct.pack("<Q", 793_100_000)
        d += struct.pack("<Q", 0)
        d += struct.pack("<Q", 1_000_000_000)
        d += b"\x00"  # complete=False (offset now 49)
        d += bytes(creator) if creator is not None else b"\x00" * 24  # 81 vs 73 bytes
        return d

    def test_creator_parsed(self) -> None:
        state = PumpFunProgram().decode_bonding_curve(self._curve(CREATOR))
        assert state is not None
        assert state.creator == str(CREATOR)

    def test_creator_absent_when_short(self) -> None:
        state = PumpFunProgram().decode_bonding_curve(self._curve(None))
        assert state is not None
        assert state.creator is None

    def test_dataclass_default_creator_none(self) -> None:
        assert (
            BondingCurveState(
                virtual_token_reserves=1,
                virtual_sol_reserves=1,
                real_token_reserves=1,
                real_sol_reserves=0,
                token_total_supply=1,
                complete=False,
            ).creator
            is None
        )
