# orb_15min_retest_from_parquet.py

import warnings
warnings.filterwarnings("ignore")

import os, glob, time
from datetime import timedelta
from typing import List, Tuple
import numpy as np
import pandas as pd

# =============================
# USER PARAMETERS
# =============================

# Parquet directory (ALL files inside will be processed)
PARQUET_DIR = r"C:\Users\pcagm\OneDrive\Desktop\downloads\parquet\parquet"

# If timestamps in your parquet are NAIVE and represent UTC, keep True
ASSUME_NAIVE_TIMESTAMPS_ARE_UTC = True

# Session / timezone
TZ = "America/New_York"
REG_SESSION_START = "09:30"
REG_SESSION_END   = "16:00"

# ORB logic (IDENTICAL BEHAVIOR)
RETEST_CONFIRM_CLOSE = True          # retest bar must CLOSE back through the level
MAX_RETEST_MIN = 120                 # minutes after breakout to wait for retest

# Risk / exits (IDENTICAL)
R_MULTIPLES = [1.0, 2.0]             # 1R and 2R targets
POSITION_SIZE_DOLLARS = 10_000
SLIPPAGE_BPS = 1.0
FEES_PER_TRADE = 0.00

# Batch / persistence (keep the autosave cadence)
AUTOSAVE_EVERY = 25
SLEEP_BETWEEN_TICKERS = 0.2

# Output files (same filenames for continuity)
TRADES_CSV_ALL = "orb_trades_sp500.csv"
EQUITY_CSV_ALL = "orb_equity_sp500.csv"

# =============================
# Helpers
# =============================
STD_MAP = {"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}

def _standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    lower = {c.lower(): c for c in df.columns}
    for k, std in STD_MAP.items():
        if k in lower:                      rename[lower[k]] = std
        elif k.capitalize() in df.columns:  rename[k.capitalize()] = std
        elif k.upper() in df.columns:       rename[k.upper()] = std
    out = df.rename(columns=rename)
    drop_cols = [c for c in out.columns if str(c).lower().startswith("adj")]
    return out.drop(columns=drop_cols, errors="ignore")

def _choose_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    for cand in ["EventAt","Datetime","datetime","Timestamp","timestamp","Date","date","Time","time"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand], errors="coerce")
            df = df.set_index(cand)
            break
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("No datetime index/column found.")
    return df

def _infer_median_interval_minutes(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2: return np.nan
    d = (idx[1:] - idx[:-1]).total_seconds() / 60.0
    return float(np.median(d)) if len(d) else np.nan

def _resample_to_15m_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match Yahoo 15m behavior:
      - If bars are finer than 15m, resample to exact 15m OHLC(V)
        using right-closed, right-labeled bins so the first bar is 09:45.
      - If already ~15m, keep as-is.
      - If coarser than 15m, skip (return empty for this strategy).
    """
    m = _infer_median_interval_minutes(df.index)
    if np.isnan(m): return pd.DataFrame()
    if m < 15 - 1e-6:
        agg = {"Open":"first","High":"max","Low":"min","Close":"last"}
        if "Volume" in df.columns: agg["Volume"] = "sum"
        out = (df.sort_index()
                 .resample("15T", label="right", closed="right")
                 .agg(agg)
                 .dropna(subset=["Open","High","Low","Close"]))
        return out
    elif abs(m - 15) <= 1e-6:
        return df.sort_index()
    else:
        return pd.DataFrame()  # coarser than 15m → not usable here

def load_parquet_15m(path: str, tz: str = TZ) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    df = _choose_dt_index(df).sort_index()

    # TZ handling
    if df.index.tz is None:
        if ASSUME_NAIVE_TIMESTAMPS_ARE_UTC:
            df = df.tz_localize("UTC").tz_convert(tz)
        else:
            df = df.tz_localize(tz)
    else:
        df = df.tz_convert(tz)

    df = _standardize_ohlcv_columns(df)
    for need in ["Open","High","Low","Close"]:
        if need not in df.columns:
            raise ValueError(f"{os.path.basename(path)} missing required column: {need}")

    df = _resample_to_15m_if_needed(df)
    if df.empty:
        return df

    # Regular session only (matches the original)
    df = df.between_time(REG_SESSION_START, REG_SESSION_END)
    return df.sort_index()

def sessionize(df_15: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df_15.index.date)

# =============================
# ORB logic 
# =============================
def compute_opening_range(df_15: pd.DataFrame) -> pd.DataFrame:
    df = df_15.copy()

    # Ensure OR columns exist
    for col in ["ORH", "ORL"]:
        if col not in df.columns:
            df[col] = np.nan

    df["Session"] = sessionize(df)

    # First 15m bar per day
    first_bar_idx = df.groupby("Session").head(1).index
    df.loc[first_bar_idx, "ORH"] = df.loc[first_bar_idx, "High"].astype(float)
    df.loc[first_bar_idx, "ORL"] = df.loc[first_bar_idx, "Low"].astype(float)

    # Forward-fill within session
    df[["ORH","ORL"]] = df.groupby("Session")[["ORH","ORL"]].ffill()
    return df

def _apply_slippage(price: float, bps: float, side: str) -> float:
    factor = 1 + (bps/10000.0)
    return price * factor if side == "buy" else price / factor

def backtest_orb_retest(
    df_15: pd.DataFrame,
    max_retest_min: int = MAX_RETEST_MIN,
    r_targets: List[float] = R_MULTIPLES,
    dollars: float = POSITION_SIZE_DOLLARS,
    slippage_bps: float = SLIPPAGE_BPS,
    fees: float = FEES_PER_TRADE,
    retest_confirm_close: bool = RETEST_CONFIRM_CLOSE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exact behavior from orb_15min_retest_sp500.py:
      - Opening range = first 15m bar high/low
      - First breakout CLOSE beyond ORH/ORL
      - Wait up to MAX_RETEST_MIN for retest
      - If RETEST_CONFIRM_CLOSE=True, retest bar must CLOSE back across the level
      - Entry at the OR level (with slippage); stop = opposite OR
      - Exits: stop FIRST, else first TP hit (1R then 2R), else EOD
      - PnL splits size equally among realized exit legs
    """
    df = df_15.copy()
    df["Session"] = sessionize(df)
    sessions = df["Session"].unique()

    trades = []
    for ses in sessions:
        sdf = df[df["Session"] == ses].copy()
        if len(sdf) < 3:
            continue

        orh = sdf["ORH"].iloc[0]
        orl = sdf["ORL"].iloc[0]
        if pd.isna(orh) or pd.isna(orl):
            continue

        # First breakout CLOSE after the opening bar
        sdf_after = sdf.iloc[1:].copy()
        long_break  = sdf_after[sdf_after["Close"] > orh].head(1)
        short_break = sdf_after[sdf_after["Close"] < orl].head(1)

        if long_break.empty and short_break.empty:
            continue

        if not long_break.empty and not short_break.empty:
            direction = "long" if long_break.index[0] < short_break.index[0] else "short"
            breakout_row = long_break.iloc[0] if direction == "long" else short_break.iloc[0]
        elif not long_break.empty:
            direction = "long"; breakout_row = long_break.iloc[0]
        else:
            direction = "short"; breakout_row = short_break.iloc[0]

        # Retest within window
        btime = breakout_row.name
        cutoff = btime + timedelta(minutes=max_retest_min)
        after_break = sdf[(sdf.index > btime) & (sdf.index <= cutoff)].copy()
        if after_break.empty:
            continue

        level = orh if direction == "long" else orl
        touch = after_break[(after_break["Low"] <= level) & (after_break["High"] >= level)].head(1)
        if touch.empty:
            continue

        retest_bar = touch.iloc[0]
        retest_time = retest_bar.name

        if retest_confirm_close:
            ok = (retest_bar["Close"] >= level) if direction == "long" else (retest_bar["Close"] <= level)
            if not ok:
                continue

        entry = _apply_slippage(level, slippage_bps, "buy" if direction == "long" else "sell")
        stop = orl if direction == "long" else orh
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 1e-12:
            continue

        qty = dollars / entry
        side_mult = 1 if direction == "long" else -1

        # Target prices (1R, 2R, ...)
        t_prices = [(entry + r*risk_per_share) if direction == "long" else (entry - r*risk_per_share) for r in r_targets]

        # Forward simulate exits
        sdf_run = sdf[sdf.index >= retest_time].copy()
        exits = []
        for ts, row in sdf_run.iterrows():
            high, low = float(row["High"]), float(row["Low"])
            # Stop first
            if low <= stop <= high:
                px = _apply_slippage(stop, slippage_bps, "sell" if direction == "long" else "buy")
                exits.append(("stop", ts, px))
                break
            # Targets (first hit wins)
            hit = False
            for i, tp in enumerate(t_prices):
                if low <= tp <= high:
                    px = _apply_slippage(tp, slippage_bps, "sell" if direction == "long" else "buy")
                    exits.append((f"tp{i+1}", ts, px))
                    hit = True
            if hit:
                break

        if not exits:
            # Flat on last bar close (EOD)
            last = sdf_run.iloc[-1]
            px = _apply_slippage(float(last["Close"]), slippage_bps, "sell" if direction == "long" else "buy")
            exits.append(("eod", sdf_run.index[-1], px))

        # Equal partials among realized exits
        n_parts = len(exits)
        qty_each = qty / n_parts
        cash_pnl = 0.0
        for tag, ts, px in exits:
            cash_pnl += (px - entry) * side_mult * qty_each
        cash_pnl -= FEES_PER_TRADE

        trades.append({
            "Ticker": sdf["Ticker"].iloc[0] if "Ticker" in sdf.columns else "",
            "Session": ses,
            "ORH": float(orh), "ORL": float(orl),
            "Direction": direction,
            "EntryTime": retest_time,
            "Entry": float(entry), "Stop": float(stop),
            "Targets": [float(x) for x in t_prices],
            "Exits": [(str(t[0]), t[1], float(t[2])) for t in exits],
            "Qty": float(qty),
            "PnL_$": float(cash_pnl),
            "R_multiple": float(cash_pnl / (risk_per_share * qty))
        })

    trade_log = pd.DataFrame(trades)
    if trade_log.empty:
        eq = pd.DataFrame(columns=["Session","CumPnL_$"])
        return trade_log, eq

    equity = (trade_log.groupby("Session")["PnL_$"].sum()
              .sort_index().cumsum().reset_index().rename(columns={"PnL_$":"CumPnL_$"}))
    return trade_log, equity

# =============================
# Runner (folder of parquet files)
# =============================
def run_folder(folder: str):
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    print(f"Found {len(files)} parquet files.")
    all_trades, all_equity = [], []

    for i, fpath in enumerate(files, 1):
        ticker = os.path.splitext(os.path.basename(fpath))[0].upper().replace("_","").replace("-","")
        print(f"\n== {ticker} ({i}/{len(files)}) ==\n{fpath}")
        try:
            df15 = load_parquet_15m(fpath, tz=TZ)
            if df15.empty:
                print("Skipped (no 15m data after normalization/resample/session filter).")
                continue
            df15["Ticker"] = ticker
            df15 = compute_opening_range(df15)
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue

        tlog, eq = backtest_orb_retest(
            df15,
            max_retest_min=MAX_RETEST_MIN,
            r_targets=R_MULTIPLES,
            dollars=POSITION_SIZE_DOLLARS,
            slippage_bps=SLIPPAGE_BPS,
            fees=FEES_PER_TRADE,
            retest_confirm_close=RETEST_CONFIRM_CLOSE
        )

        if not tlog.empty:
            out = tlog.copy()
            out["Targets"] = out["Targets"].apply(lambda xs: ";".join(f"{p:.6f}" for p in xs))
            out["Exits"] = out["Exits"].apply(lambda xs: ";".join(f"{t}|{ts}|{px:.6f}" for (t, ts, px) in xs))
            all_trades.append(out)
        if not eq.empty:
            eq2 = eq.copy(); eq2["Ticker"] = ticker
            all_equity.append(eq2)

        if AUTOSAVE_EVERY and i % AUTOSAVE_EVERY == 0:
            if all_trades:
                pd.concat(all_trades, ignore_index=True).to_csv(TRADES_CSV_ALL, index=False)
                print(f"[autosave] wrote {TRADES_CSV_ALL}")
            if all_equity:
                pd.concat(all_equity, ignore_index=True).to_csv(EQUITY_CSV_ALL, index=False)
                print(f"[autosave] wrote {EQUITY_CSV_ALL}")

        time.sleep(SLEEP_BETWEEN_TICKERS)

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()

    if not trades.empty:
        trades.sort_values(["Ticker","Session"], inplace=True)
        trades.to_csv(TRADES_CSV_ALL, index=False)
        print(f"Saved trades: {TRADES_CSV_ALL}")

    if not equity.empty:
        equity.sort_values(["Ticker","Session"], inplace=True)
        equity.to_csv(EQUITY_CSV_ALL, index=False)
        print(f"Saved equity: {EQUITY_CSV_ALL}")

    return trades, equity

# =============================
# Main
# =============================
def main():
    trades, equity = run_folder(PARQUET_DIR)

    # Overall totals (same style as original)
    if trades is not None and not trades.empty:
        total_trades = len(trades)
        total_pnl    = trades["PnL_$"].sum()
        win_rate     = (trades["PnL_$"] > 0).mean() * 100.0
        avg_r        = trades["R_multiple"].mean()
        median_r     = trades["R_multiple"].median()

        print("\n=== Overall Totals ===")
        print(f"Total trades: {total_trades}")
        print(f"Total PnL ($): {total_pnl:,.2f}")
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Avg R: {avg_r:.3f} | Median R: {median_r:.3f}")
    else:
        print("No trades generated with current settings.")

if __name__ == "__main__":
    main()
