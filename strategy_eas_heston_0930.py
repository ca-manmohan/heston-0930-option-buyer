"""
EAS-Heston-0930 Execution Engine (Rs 20,000 Base Capital Edition)
=================================================================
Production Option-Buying Engine for the 09:30:00 IST Weekly-vs.-Monthly Heston
Term-Structure Breakdown & Vilkov GEX Strategy.

Key Specifications:
  - Instrument: NIFTY 50 Weekly ATM Option Buying ONLY (CE or PE, Lot Size = 75)
  - Entry Timestamp: Strictly 09:30:00 IST (Zero Lookahead — uses 09:15-09:30 IST live
    Heston calibration, Vilkov Semivariance Skew & GEX flow, and t-1 FII LSR_prev)
  - Dynamic Position Sizing (Rs 20,000 Base Capital):
      * Minimum 2 lots (150 qty) to enable 50/50 Dual-Tranche split
      * Scales dynamically up to 6 lots (450 qty) based on available equity and option premium
  - Dual-Tranche High-Convexity Exit Engine (09:30:00 -> 11:30:00 IST):
      * Tranche 1 (50% lots): Quick-harvest limit target at +12.0 option points
      * Tranche 2 (50% lots):
          - Once Tranche 1 hits, stop moves to Breakeven (+0.5 pts) on the next 1-minute bar
          - Once peak option gain reaches >= +35.0 pts, activates a 0.65 * prev_peak trailing ratchet
          - Mandatory time stop at 11:30:00 IST if not ratcheted
      * Hard Stop-Loss (before Tranche 1 hits): -16.0 option points on full position
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
from symbolic_heston_0930_invariant import LimiteHeston0930Invariant


@dataclass
class HestonPositionSizingPlan:
    total_lots: int
    tranche1_lots: int
    tranche2_lots: int
    total_qty: int
    margin_deployed_inr: float
    max_risk_inr: float


class EASHeston0930Engine:
    """
    Production execution and lifecycle manager for EAS-Heston-0930.
    """

    LOT_SIZE = 75
    MIN_LOTS = 2
    MAX_LOTS = 6
    FRICTION_PER_LOT_INR = 40.0

    T1_TARGET_PTS = 12.0
    HARD_SL_PTS = -16.0
    BREAKEVEN_BUFFER_PTS = 0.5
    RATCHET_ACTIVATION_PTS = 35.0
    RATCHET_RETENTION_RATIO = 0.65

    def __init__(self, initial_capital: float = 20000.0):
        self.initial_capital = float(initial_capital)
        self.current_capital = float(initial_capital)
        self.symbolic = LimiteHeston0930Invariant()

    def compute_lots(self, option_entry_price: float) -> HestonPositionSizingPlan:
        cost_per_lot = max(45.0, float(option_entry_price)) * self.LOT_SIZE
        raw_affordable = int((self.current_capital * 0.95) // cost_per_lot)
        even_lots = max(self.MIN_LOTS, min(self.MAX_LOTS, (raw_affordable // 2) * 2))
        if even_lots < self.MIN_LOTS:
            even_lots = self.MIN_LOTS

        t1_lots = even_lots // 2
        t2_lots = even_lots - t1_lots
        total_qty = even_lots * self.LOT_SIZE
        margin_used = even_lots * cost_per_lot
        max_risk = total_qty * abs(self.HARD_SL_PTS) + even_lots * self.FRICTION_PER_LOT_INR

        return HestonPositionSizingPlan(
            total_lots=even_lots,
            tranche1_lots=t1_lots,
            tranche2_lots=t2_lots,
            total_qty=total_qty,
            margin_deployed_inr=round(margin_used, 2),
            max_risk_inr=round(max_risk, 2),
        )

    def evaluate_0930_gate(
        self,
        heston_wm_vol_ratio: float,
        iv_rmse_w_pct: float,
        r_gamma_net: float,
        b_gamma: float,
        vilkov_is_skew: float,
        term_basis_spread_wm: float,
        atm_strike: int,
        ce_entry_price: float,
        pe_entry_price: float,
        transfer_ratio_mw: float = 1.0,
        pcp_abs_dev_pct: float = 0.0,
        lsr_prev_pct: float = 11.0,
    ) -> Dict[str, Any]:
        sym = self.symbolic.evaluate(
            heston_wm_vol_ratio=heston_wm_vol_ratio,
            iv_rmse_w_pct=iv_rmse_w_pct,
            r_gamma_net=r_gamma_net,
            b_gamma=b_gamma,
            vilkov_is_skew=vilkov_is_skew,
            term_basis_spread_wm=term_basis_spread_wm,
            transfer_ratio_mw=transfer_ratio_mw,
            pcp_abs_dev_pct=pcp_abs_dev_pct,
            lsr_prev_pct=lsr_prev_pct,
        )

        if sym["decision"] == "BUY_ATM_CE":
            opt_type = "CE"
            entry_px = float(ce_entry_price)
        elif sym["decision"] == "BUY_ATM_PE":
            opt_type = "PE"
            entry_px = float(pe_entry_price)
        else:
            return {
                **sym,
                "order_status": "NO_TRADE",
                "instrument": None,
                "entry_price": 0.0,
                "sizing": None,
            }

        sizing = self.compute_lots(entry_px)
        return {
            **sym,
            "order_status": "EXECUTE_0930_IST",
            "instrument": f"NIFTY {atm_strike} {opt_type}",
            "option_type": opt_type,
            "atm_strike": int(atm_strike),
            "entry_price": round(entry_px, 2),
            "targets": {
                "tranche1_target_price": round(entry_px + self.T1_TARGET_PTS, 2),
                "hard_stop_price": round(max(0.05, entry_px + self.HARD_SL_PTS), 2),
                "ratchet_activation_price": round(entry_px + self.RATCHET_ACTIVATION_PTS, 2),
                "time_exit_ist": "11:30:00",
            },
            "sizing": sizing.__dict__,
        }

    def simulate_dual_tranche_path(
        self,
        entry_price: float,
        bars_1m: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        t1_done = False
        t1_pts = 0.0
        t2_done = False
        t2_pts = 0.0
        peak_pts = 0.0
        exit_reason = "TIME_1130"

        for b in bars_1m:
            hi_pts = float(b["high"]) - entry_price
            lo_pts = float(b["low"]) - entry_price

            prev_peak = peak_pts
            if hi_pts > peak_pts:
                peak_pts = hi_pts

            if not t1_done:
                if lo_pts <= self.HARD_SL_PTS:
                    t1_pts = self.HARD_SL_PTS
                    t2_pts = self.HARD_SL_PTS
                    t1_done = True
                    t2_done = True
                    exit_reason = "HARD_SL"
                    break
                if hi_pts >= self.T1_TARGET_PTS:
                    t1_pts = self.T1_TARGET_PTS
                    t1_done = True
                    continue
            else:
                trail_stop = self.BREAKEVEN_BUFFER_PTS
                if prev_peak >= self.RATCHET_ACTIVATION_PTS:
                    trail_stop = max(trail_stop, prev_peak * self.RATCHET_RETENTION_RATIO)
                if lo_pts <= trail_stop:
                    t2_pts = trail_stop
                    t2_done = True
                    exit_reason = (
                        "RATCHET"
                        if prev_peak >= self.RATCHET_ACTIVATION_PTS
                        else "BREAKEVEN"
                    )
                    break

        if not t1_done:
            last_cl = float(bars_1m[-1]["close"]) - entry_price if bars_1m else 0.0
            t1_pts = last_cl
            t2_pts = last_cl
        elif not t2_done:
            last_cl = float(bars_1m[-1]["close"]) - entry_price if bars_1m else 0.0
            t2_pts = last_cl

        blended_pts = 0.5 * t1_pts + 0.5 * t2_pts
        return {
            "t1_pts": round(t1_pts, 2),
            "t2_pts": round(t2_pts, 2),
            "blended_pts": round(blended_pts, 2),
            "peak_pts": round(peak_pts, 2),
            "exit_reason": exit_reason,
        }
