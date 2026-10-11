"""
Empirical Research & Feature Matrix Builder:
Strategy 2 — Weekly-vs.-Monthly Heston Term-Structure Breakdown ("Stress Persistence") Option Buyer
===================================================================================================
Synthesizes:
  1. Manivanan & Pandey (Sep 2026): Weekly (kappa_W=15.72, theta_W=0.0236, xi_W=1.04, rho_W=-0.425)
     vs. Monthly (kappa_M=6.40, theta_M=0.0987, xi_M=1.79, rho_M=-0.280) Heston calibration,
     smile IV_RMSE, and Cross-Structure Transfer Penalty across all 9-18 Weekly strikes.
  2. Priyan & Aggarwal (IIM Udaipur Dec 2025): Post-SEBI Put-Call Parity Absolute Deviation %
     and Weekly Synthetic Basis vs. Monthly Futures Basis term-structure dislocation.
  3. Grigory Vilkov (Mar 2026): Up-minus-Down OTM Implied Semivariance Skew (IS_t = IV_up - IV_dn),
     Normalized OI Gamma Balance (B_Gamma), and Traded Gamma Flow Pressure over OI Base (R_Gamma).
  4. Zero-Lookahead t-1 FII Stress Persistence Telemetry (LSR_prev, FII_Cash_prev, Prev_Day_Return).
"""

from __future__ import annotations
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data_cache"
OPT_DIR = CACHE / "options"
OUT_DIR = ROOT / "backtest_results" / "heston_stress_persistence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Manivanan & Pandey (2026) Table 2 & Table 5 structural Heston benchmarks
RISK_FREE_RATE = 0.0530  # 91-day GOI T-bill yield proxy (Table 2 mean ~5.31%)
HESTON_WEEKLY_BENCH = {"kappa": 15.7197, "theta": 0.0236, "xi": 1.0360, "rho": -0.4252}
HESTON_MONTHLY_BENCH = {"kappa": 6.4030, "theta": 0.0987, "xi": 1.7856, "rho": -0.2805}


def load_1m_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw:
        return pd.DataFrame()
    all_cols = ["dt", "open", "high", "low", "close", "volume", "oi"]
    ncols = len(raw[0])
    df = pd.DataFrame(raw, columns=all_cols[:ncols])
    if "oi" not in df.columns:
        df["oi"] = 0.0
    df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize(None)
    df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
    df["time_str"] = df.index.strftime("%H:%M")
    return df


def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    if T <= 1e-6 or sigma <= 1e-4:
        return max(0.0, S - K) if opt_type == "CE" else max(0.0, K - S)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if opt_type == "CE":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 1e-6 or sigma <= 1e-4:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return S * sqrtT * norm.pdf(d1)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 1e-6 or sigma <= 1e-4:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return norm.pdf(d1) / (S * sigma * sqrtT)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    if T <= 1e-6 or sigma <= 1e-4:
        if opt_type == "CE":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return norm.cdf(d1) if opt_type == "CE" else norm.cdf(d1) - 1.0


def implied_vol(price: float, S: float, K: float, T: float, r: float, opt_type: str) -> float:
    intrinsic = max(0.0, S - K * math.exp(-r * T)) if opt_type == "CE" else max(0.0, K * math.exp(-r * T) - S)
    if price <= intrinsic + 0.05:
        return np.nan
    low, high = 0.03, 1.50
    for _ in range(32):
        mid = 0.5 * (low + high)
        val = bs_price(S, K, T, r, mid, opt_type)
        if val > price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def heston_fjl_implied_vol(k_log: float, T: float, v0: float, kappa: float, theta: float, xi: float, rho: float) -> float:
    """
    Forde-Jacquier-Lee (2012) & Gatheral (2006) Heston implied variance expansion
    for short-to-medium maturities (cited in Manivanan & Pandey 2026 Section 'Maturity and IV Structure').
    Captures exact Heston mean-reversion (kappa, theta), spot-vol correlation skew (rho*xi),
    and vol-of-vol smile curvature (xi^2 * (1 - rho^2)).
    """
    tau = max(1.0 / 365.0, float(T))
    # Expected integrated variance under Heston (exact)
    decay = (1.0 - math.exp(-kappa * tau)) / (kappa * tau) if kappa * tau > 1e-5 else 1.0
    v_bar = theta + (v0 - theta) * decay
    v_bar = max(1e-4, v_bar)
    sigma_0 = math.sqrt(v_bar)

    # First-order skew (spot-vol correlation rho * xi) and second-order curvature (xi^2)
    skew_term = (rho * xi / (2.0 * sigma_0)) * (1.0 - 0.25 * kappa * tau / (1.0 + 0.5 * kappa * tau))
    curv_term = (xi * xi / (12.0 * max(1e-3, v_bar))) * (1.0 - 1.25 * rho * rho)

    sigma_k = sigma_0 * (1.0 + skew_term * k_log + curv_term * (k_log ** 2))
    return max(0.03, min(1.50, sigma_k))


def calibrate_slice_heston_and_transfer(
    k_logs: np.ndarray,
    market_ivs: np.ndarray,
    vegas: np.ndarray,
    T: float,
    v0_init: float,
) -> Dict[str, float]:
    """
    Calibrates Heston parameters (v0, kappa, theta, xi, rho) on the observed option slice
    using Manivanan & Pandey's vega-weighted squared IV objective (Eq. on page 9),
    and evaluates both Weekly-benchmark vs Monthly-benchmark cross-structure transfer penalties.
    """
    w = vegas / max(1e-6, np.sum(vegas))

    def loss_fn(params):
        v0, kappa, theta, xi, rho = params
        model_ivs = np.array([
            heston_fjl_implied_vol(k, T, v0, kappa, theta, xi, rho) for k in k_logs
        ])
        return float(np.sum(w * ((model_ivs - market_ivs) ** 2)))

    # Evaluate native Weekly Benchmark vs Monthly Benchmark (fixing v0 to ATM variance)
    loss_weekly_bench = loss_fn([
        v0_init,
        HESTON_WEEKLY_BENCH["kappa"],
        HESTON_WEEKLY_BENCH["theta"],
        HESTON_WEEKLY_BENCH["xi"],
        HESTON_WEEKLY_BENCH["rho"],
    ])
    loss_monthly_bench = loss_fn([
        v0_init,
        HESTON_MONTHLY_BENCH["kappa"],
        HESTON_MONTHLY_BENCH["theta"],
        HESTON_MONTHLY_BENCH["xi"],
        HESTON_MONTHLY_BENCH["rho"],
    ])

    # Optimize slice parameters within Table 3 bounds
    bounds = [
        (0.001, 0.250),   # v0
        (0.100, 20.000),  # kappa
        (0.001, 0.250),   # theta
        (0.050, 5.000),   # xi
        (-0.990, 0.990),  # rho
    ]
    x0 = [
        min(0.24, max(0.002, v0_init)),
        HESTON_WEEKLY_BENCH["kappa"],
        HESTON_WEEKLY_BENCH["theta"],
        HESTON_WEEKLY_BENCH["xi"],
        HESTON_WEEKLY_BENCH["rho"],
    ]
    res = minimize(loss_fn, x0, method="L-BFGS-B", bounds=bounds)
    v0_opt, kappa_opt, theta_opt, xi_opt, rho_opt = res.x
    model_ivs_opt = np.array([
        heston_fjl_implied_vol(k, T, v0_opt, kappa_opt, theta_opt, xi_opt, rho_opt) for k in k_logs
    ])
    iv_rmse = float(np.sqrt(np.mean((model_ivs_opt - market_ivs) ** 2)))

    # Expected variance ratio between Monthly Heston benchmark and Weekly Heston benchmark
    tau = max(1.0 / 365.0, float(T))
    decay_w = (1.0 - math.exp(-HESTON_WEEKLY_BENCH["kappa"] * tau)) / (HESTON_WEEKLY_BENCH["kappa"] * tau)
    decay_m = (1.0 - math.exp(-HESTON_MONTHLY_BENCH["kappa"] * tau)) / (HESTON_MONTHLY_BENCH["kappa"] * tau)
    exp_var_w = HESTON_WEEKLY_BENCH["theta"] + (v0_init - HESTON_WEEKLY_BENCH["theta"]) * decay_w
    exp_var_m = HESTON_MONTHLY_BENCH["theta"] + (v0_init - HESTON_MONTHLY_BENCH["theta"]) * decay_m
    heston_wm_var_ratio = math.sqrt(max(1e-4, exp_var_w) / max(1e-4, exp_var_m))

    # Transfer ratio: Monthly benchmark loss vs Weekly benchmark loss on the current slice
    transfer_ratio_mw = loss_monthly_bench / max(1e-8, loss_weekly_bench)

    return {
        "v0_opt": float(v0_opt),
        "kappa_opt": float(kappa_opt),
        "theta_opt": float(theta_opt),
        "xi_opt": float(xi_opt),
        "rho_opt": float(rho_opt),
        "iv_rmse_w": iv_rmse,
        "loss_weekly_bench": float(loss_weekly_bench),
        "loss_monthly_bench": float(loss_monthly_bench),
        "transfer_ratio_mw": float(transfer_ratio_mw),
        "heston_wm_vol_ratio": float(heston_wm_var_ratio),
    }


def simulate_dual_tranche_option(
    df_opt: pd.DataFrame,
    entry_time: str,
    exit_time: str,
    t1_target: float = 12.0,
    hard_sl: float = -16.0,
    ratchet_trigger: float = 40.0,
    ratchet_keep: float = 0.65,
) -> Dict[str, Any]:
    sub = df_opt[(df_opt["time_str"] >= entry_time) & (df_opt["time_str"] <= exit_time)]
    if len(sub) < 2:
        return {"entry_px": np.nan, "blended_pts": 0.0, "peak_pts": 0.0, "exit_reason": "NO_DATA"}

    entry_px = float(sub.iloc[0]["close"])
    bars = sub.iloc[1:]

    t1_done = False
    t1_pts = 0.0
    t2_done = False
    t2_pts = 0.0
    peak_pts = 0.0
    exit_reason = f"TIME_{exit_time.replace(':', '')}"

    for _, row in bars.iterrows():
        hi_pts = float(row["high"]) - entry_px
        lo_pts = float(row["low"]) - entry_px
        cl_pts = float(row["close"]) - entry_px

        prev_peak = peak_pts
        if hi_pts > peak_pts:
            peak_pts = hi_pts

        if not t1_done:
            if lo_pts <= hard_sl:
                t1_pts = hard_sl
                t2_pts = hard_sl
                t1_done = True
                t2_done = True
                exit_reason = "HARD_SL"
                break
            if hi_pts >= t1_target:
                t1_pts = t1_target
                t1_done = True

        if t1_done and not t2_done:
            trail = 0.5
            if prev_peak >= ratchet_trigger:
                trail = max(trail, prev_peak * ratchet_keep)
            if lo_pts <= trail and cl_pts < hi_pts:
                t2_pts = trail
                t2_done = True
                exit_reason = "RATCHET" if prev_peak >= ratchet_trigger else "BREAKEVEN"
                break

    if not t1_done:
        last_cl = float(bars.iloc[-1]["close"]) - entry_px
        t1_pts = last_cl
        t2_pts = last_cl
    elif not t2_done:
        last_cl = float(bars.iloc[-1]["close"]) - entry_px
        t2_pts = last_cl

    return {
        "entry_px": round(entry_px, 2),
        "t1_pts": round(t1_pts, 2),
        "t2_pts": round(t2_pts, 2),
        "blended_pts": round(0.5 * (t1_pts + t2_pts), 2),
        "peak_pts": round(peak_pts, 2),
        "exit_reason": exit_reason,
    }


def build_session_dataset():
    # Load FII synthesis table and shift by 1 session for strict zero-lookahead (t-1)
    fii_df = pd.read_csv(CACHE / "fii" / "fii_daily_synthesis_june_to_oct.csv")
    fii_df["date"] = pd.to_datetime(fii_df["date"]).dt.strftime("%Y-%m-%d")
    fii_df = fii_df.sort_values("date").reset_index(drop=True)
    fii_df["lsr_prev"] = fii_df["fii_lsr_pct"].shift(1).bfill()
    fii_df["fii_cash_prev"] = fii_df["fpi_net"].shift(1).fillna(0.0)
    fii_df["fii_fut_chg_prev"] = fii_df["fii_net_fut"].diff().shift(1).fillna(0.0)
    fii_map = fii_df.set_index("date").to_dict(orient="index")

    # Index all option files by (date, expiry, strike, opt_type)
    opt_index: Dict[Tuple[str, str], Dict[Tuple[int, str], Path]] = {}
    for f in OPT_DIR.glob("NSE_NIFTY_*.json"):
        m = re.match(r"NSE_NIFTY_([0-9A-Za-z]+)_(\d+)_(CE|PE)_(\d{4}-\d{2}-\d{2})\.json", f.name)
        if m:
            exp, strike, otype, dt = m.groups()
            opt_index.setdefault((dt, exp), {})[(int(strike), otype)] = f

    # Group by date and choose nearest active weekly expiry
    dates_map: Dict[str, List[str]] = {}
    for dt, exp in opt_index.keys():
        dates_map.setdefault(dt, []).append(exp)

    rows = []
    prev_day_ret_pct = 0.0
    prev_day_stress_flag = 0

    for dt in sorted(dates_map.keys()):
        spot_path = CACHE / f"spot_{dt}.json"
        fut_path = CACHE / f"fut_{dt}.json"
        if not spot_path.exists() or not fut_path.exists():
            continue

        df_spot = load_1m_json(spot_path)
        df_fut = load_1m_json(fut_path)
        if df_spot.empty or df_fut.empty:
            continue

        # Pick expiry with >= 1 DTE if multiple exist
        exps = sorted(dates_map[dt])
        chosen_exp = exps[0]
        dt_obj = datetime.strptime(dt, "%Y-%m-%d")
        exp_obj = datetime.strptime(chosen_exp, "%d%b%y")
        if exp_obj < dt_obj and len(exps) > 1:
            chosen_exp = exps[-1]
            exp_obj = datetime.strptime(chosen_exp, "%d%b%y")
        if exp_obj < dt_obj:
            # Skip mislabeled post-expiry file (e.g. 2026-10-01 having 29Sep26)
            continue

        dte_days = max(0.35, (exp_obj - dt_obj).days + 0.5)
        T_years = dte_days / 365.0

        # Extract 09:15 -> 09:30 IST morning telemetry (strictly causal at 09:30:00 IST)
        s_0915_0930 = df_spot[(df_spot["time_str"] >= "09:15") & (df_spot["time_str"] <= "09:30")]
        f_0915_0930 = df_fut[(df_fut["time_str"] >= "09:15") & (df_fut["time_str"] <= "09:30")]
        if len(s_0915_0930) < 10 or len(f_0915_0930) < 10:
            continue

        spot_open = float(s_0915_0930.iloc[0]["open"])
        spot_0930 = float(s_0915_0930.iloc[-1]["close"])
        fut_0930 = float(f_0915_0930.iloc[-1]["close"])
        spot_close_eod = float(df_spot.iloc[-1]["close"])

        # Morning 15-min impulse and Z-score
        r_0930_pct = (spot_0930 - spot_open) / spot_open * 100.0
        s_1m_rets = s_0915_0930["close"].pct_change().dropna()
        vol_1m = float(s_1m_rets.std()) * math.sqrt(15.0) * 100.0
        z_0930 = r_0930_pct / max(0.08, vol_1m)

        # Futures basis at 09:30 vs 09:15 open
        fut_basis_0915 = float(f_0915_0930.iloc[0]["close"]) - float(s_0915_0930.iloc[0]["close"])
        fut_basis_0930 = fut_0930 - spot_0930
        fut_basis_pct = fut_basis_0930 / spot_0930 * 100.0
        d_fut_basis_0930 = fut_basis_0930 - fut_basis_0915

        # Load all available strikes for (dt, chosen_exp)
        strike_files = opt_index[(dt, chosen_exp)]
        all_strikes = sorted({k[0] for k in strike_files.keys()})
        atm_strike = min(all_strikes, key=lambda x: abs(x - spot_0930))

        # Parse 09:30 cross-section across all strikes for Heston, Vilkov & Priyan metrics
        k_logs = []
        mkt_ivs = []
        vegas = []
        iv_ups = []
        iv_dns = []
        pcp_devs = []

        g_oi_signed = 0.0
        g_oi_abs = 0.0
        f_gamma_call = 0.0
        f_gamma_put = 0.0
        vol_ce_total = 0.0
        vol_pe_total = 0.0

        strike_dfs: Dict[Tuple[int, str], pd.DataFrame] = {}
        for (st, otype), pth in strike_files.items():
            odf = load_1m_json(pth)
            if not odf.empty:
                strike_dfs[(st, otype)] = odf

        for st in all_strikes:
            ce_df = strike_dfs.get((st, "CE"))
            pe_df = strike_dfs.get((st, "PE"))
            if ce_df is None or pe_df is None:
                continue
            ce_sub = ce_df[(ce_df["time_str"] >= "09:15") & (ce_df["time_str"] <= "09:30")]
            pe_sub = pe_df[(pe_df["time_str"] >= "09:15") & (pe_df["time_str"] <= "09:30")]
            if ce_sub.empty or pe_sub.empty:
                continue

            c_px = float(ce_sub.iloc[-1]["close"])
            p_px = float(pe_sub.iloc[-1]["close"])
            c_vol = float(ce_sub["volume"].sum())
            p_vol = float(pe_sub["volume"].sum())
            c_oi = max(0.0, float(ce_sub.iloc[-1]["oi"]))
            p_oi = max(0.0, float(pe_sub.iloc[-1]["oi"]))

            vol_ce_total += c_vol
            vol_pe_total += p_vol

            # Priyan & Aggarwal (2025) Put-Call Parity Deviation (Eq. 5.4.1)
            theo_put = max(0.0, c_px + st * math.exp(-RISK_FREE_RATE * T_years) - spot_0930)
            pcp_abs_dev_pct = abs(p_px - theo_put) / spot_0930 * 100.0
            pcp_devs.append(pcp_abs_dev_pct)

            # Implied volatility from OTM option (Call if st >= spot_0930 else Put)
            k_log = math.log(st / spot_0930)
            if st >= spot_0930:
                iv = implied_vol(c_px, spot_0930, st, T_years, RISK_FREE_RATE, "CE")
                if not np.isnan(iv):
                    iv_ups.append(iv * iv)
            else:
                iv = implied_vol(p_px, spot_0930, st, T_years, RISK_FREE_RATE, "PE")
                if not np.isnan(iv):
                    iv_dns.append(iv * iv)

            if not np.isnan(iv):
                vg = bs_vega(spot_0930, st, T_years, RISK_FREE_RATE, iv)
                gm = bs_gamma(spot_0930, st, T_years, RISK_FREE_RATE, iv)
                k_logs.append(k_log)
                mkt_ivs.append(iv)
                vegas.append(max(1.0, vg))

                # Vilkov (2026) GEX & Gamma Flow Pressure building blocks (Eqs. 15, 17, 18)
                gamma_dollar = gm * (spot_0930 ** 2) * 0.01
                g_oi_signed += (c_oi - p_oi) * gamma_dollar
                g_oi_abs += (c_oi + p_oi) * gamma_dollar
                f_gamma_call += c_vol * gamma_dollar
                f_gamma_put += p_vol * gamma_dollar

        if len(mkt_ivs) < 4 or (atm_strike, "CE") not in strike_dfs or (atm_strike, "PE") not in strike_dfs:
            continue

        # ATM Synthetic Basis vs Monthly Futures Basis
        ce_atm_df = strike_dfs[(atm_strike, "CE")]
        pe_atm_df = strike_dfs[(atm_strike, "PE")]
        ce_atm_0930 = float(ce_atm_df[ce_atm_df["time_str"] <= "09:30"].iloc[-1]["close"])
        pe_atm_0930 = float(pe_atm_df[pe_atm_df["time_str"] <= "09:30"].iloc[-1]["close"])
        synth_fut_0930 = atm_strike + ce_atm_0930 - pe_atm_0930
        synth_basis_0930 = synth_fut_0930 - spot_0930
        term_basis_spread_wm = synth_basis_0930 - fut_basis_0930  # Weekly Synth minus Monthly Fut

        # ATM IV & Heston calibration
        atm_idx = int(np.argmin(np.abs(np.array(k_logs))))
        iv_atm_w = float(mkt_ivs[atm_idx])
        v0_atm_w = iv_atm_w ** 2

        heston_fit = calibrate_slice_heston_and_transfer(
            np.array(k_logs), np.array(mkt_ivs), np.array(vegas), T_years, v0_atm_w
        )

        # Vilkov (2026) Implied Semivariance Skew IS_t = IV_up - IV_dn (Eq. 2)
        iv_up_mean = float(np.mean(iv_ups)) if iv_ups else v0_atm_w
        iv_dn_mean = float(np.mean(iv_dns)) if iv_dns else v0_atm_w
        vilkov_is_skew = (math.sqrt(iv_up_mean) - math.sqrt(iv_dn_mean)) * 100.0

        # Vilkov (2026) B_Gamma and R_Gamma (Eqs. 21, 22)
        b_gamma = g_oi_signed / (g_oi_abs + 1.0)
        r_gamma_total = (f_gamma_call + f_gamma_put) / (g_oi_abs + 1.0)
        r_gamma_net = (f_gamma_call - f_gamma_put) / (f_gamma_call + f_gamma_put + 1.0)
        pece_vol_0930 = vol_pe_total / max(1.0, vol_ce_total)

        # Simulate ATM CE and PE from 09:30 -> 11:30 IST (2-hour post-gate window)
        sim_ce = simulate_dual_tranche_option(ce_atm_df, "09:30", "11:30")
        sim_pe = simulate_dual_tranche_option(pe_atm_df, "09:30", "11:30")

        fii_info = fii_map.get(dt, {"lsr_prev": 20.0, "fii_cash_prev": 0.0, "fii_fut_chg_prev": 0.0})
        spot_post_0930_pts = spot_close_eod - spot_0930

        rows.append({
            "date": dt,
            "expiry": chosen_exp,
            "dte_days": round(dte_days, 2),
            "atm_strike": atm_strike,
            "spot_0930": round(spot_0930, 2),
            "r_0930_pct": round(r_0930_pct, 4),
            "z_0930": round(z_0930, 4),
            "prev_day_ret_pct": round(prev_day_ret_pct, 4),
            "prev_day_stress": int(prev_day_stress_flag),
            "lsr_prev": round(float(fii_info["lsr_prev"]), 2),
            "fii_cash_prev": round(float(fii_info["fii_cash_prev"]), 2),
            "fii_fut_chg_prev": round(float(fii_info["fii_fut_chg_prev"]), 1),
            # Heston Weekly vs Monthly metrics (Manivanan & Pandey 2026)
            "iv_atm_w_pct": round(iv_atm_w * 100.0, 2),
            "v0_atm_w": round(v0_atm_w, 5),
            "heston_wm_vol_ratio": round(heston_fit["heston_wm_vol_ratio"], 4),
            "kappa_opt_w": round(heston_fit["kappa_opt"], 3),
            "theta_opt_w": round(heston_fit["theta_opt"], 5),
            "xi_opt_w": round(heston_fit["xi_opt"], 3),
            "rho_opt_w": round(heston_fit["rho_opt"], 4),
            "iv_rmse_w": round(heston_fit["iv_rmse_w"], 5),
            "transfer_ratio_mw": round(heston_fit["transfer_ratio_mw"], 4),
            # SEBI Post-Reg Term-Structure & Parity metrics (Priyan & Aggarwal 2025)
            "fut_basis_0930": round(fut_basis_0930, 2),
            "d_fut_basis_0930": round(d_fut_basis_0930, 2),
            "synth_basis_0930": round(synth_basis_0930, 2),
            "term_basis_spread_wm": round(term_basis_spread_wm, 2),
            "pcp_abs_dev_pct": round(float(np.mean(pcp_devs)), 4),
            # Vilkov (2026) Semivariance Skew & Gamma Flow Pressure
            "vilkov_is_skew": round(vilkov_is_skew, 3),
            "b_gamma": round(b_gamma, 4),
            "r_gamma_total": round(r_gamma_total, 4),
            "r_gamma_net": round(r_gamma_net, 4),
            "pece_vol_0930": round(pece_vol_0930, 3),
            # Realized Option Payoffs (09:30 -> 11:30 IST)
            "ce_entry_px": sim_ce["entry_px"],
            "ce_blended_pts": sim_ce["blended_pts"],
            "ce_peak_pts": sim_ce["peak_pts"],
            "ce_exit_reason": sim_ce["exit_reason"],
            "pe_entry_px": sim_pe["entry_px"],
            "pe_blended_pts": sim_pe["blended_pts"],
            "pe_peak_pts": sim_pe["peak_pts"],
            "pe_exit_reason": sim_pe["exit_reason"],
            "spot_post_0930_pts": round(spot_post_0930_pts, 2),
        })

        # Update prev_day_ret_pct for next session (strictly causal)
        day_ret = (spot_close_eod - spot_open) / spot_open * 100.0
        prev_day_ret_pct = day_ret
        prev_day_stress_flag = 1 if abs(day_ret) >= 0.45 or float(fii_info["lsr_prev"]) < 20.0 else 0

    df_out = pd.DataFrame(rows)
    out_csv = OUT_DIR / "heston_wm_stress_persistence_56_sessions.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved {len(df_out)} sessions to {out_csv}")
    print("\nSummary of Heston Weekly vs Monthly & Vilkov Telemetry across 56 sessions:")
    cols_to_show = [
        "iv_atm_w_pct", "heston_wm_vol_ratio", "iv_rmse_w", "transfer_ratio_mw",
        "term_basis_spread_wm", "pcp_abs_dev_pct", "vilkov_is_skew", "r_gamma_net"
    ]
    print(df_out[cols_to_show].describe().round(4))


if __name__ == "__main__":
    build_session_dataset()
