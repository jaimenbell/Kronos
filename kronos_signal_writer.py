#!/usr/bin/env python3
"""
Kronos Signal Writer
Generates day-ahead price forecasts for a list of tickers and writes them
atomically to the shared signal bus at C:/Users/owner/projects/signals/kronos_predictions.json

Usage:
    python kronos_signal_writer.py --tickers MSFT AAPL TSLA
    python kronos_signal_writer.py --watchlist watchlist.txt
    python kronos_signal_writer.py --tickers MSFT --api   # use REST API if running
"""

import sys
import os
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SIGNALS_DIR = Path("C:/Users/owner/projects/signals")
OUTPUT_FILE = SIGNALS_DIR / "kronos_predictions.json"
PROJECT_ROOT = Path(__file__).parent

# Ensure `model` package is importable when running from any directory
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
LOOKBACK = 100        # trading days of history to feed the model
# 10 samples gives ±14pp standard error on win_probability — too noisy for conviction-gated entries.
# 100 samples gives ±5pp which is usable. Chronos paper uses 20 as minimum; 100 is our floor.
N_RUNS = 100          # stochastic forward passes for win-probability estimation
TEMPERATURE = 1.0     # sampling temperature (> 0 enables diversity across runs)
TOP_P = 0.9
# Model options: "kronos-mini" (4.1M), "kronos-small" (24.7M), "kronos-base" (102M)
MODEL_KEY = "kronos-small"
API_URL = "http://localhost:7070"

_MODEL_REGISTRY = {
    "kronos-mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-2k",   2048),
    "kronos-small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base",  512),
    "kronos-base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base",  512),
}


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, lookback: int) -> tuple:
    """Download daily OHLCV from Yahoo Finance. Returns (df, timestamps_series)."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is required: pip install yfinance")

    raw = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Handle yfinance MultiIndex columns (newer versions)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns=str.lower)
    required = {"open", "high", "low", "close"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")

    if "volume" not in raw.columns:
        raw["volume"] = 0.0

    raw = raw[["open", "high", "low", "close", "volume"]].dropna().tail(lookback)

    if len(raw) < 20:
        raise ValueError(f"{ticker}: only {len(raw)} bars available (need ≥ 20)")

    # Produce a tz-naive DatetimeIndex — Kronos calc_time_stamps uses dt accessors
    idx = pd.to_datetime(raw.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    raw.index = idx
    timestamps = pd.Series(idx, name="timestamp")

    return raw, timestamps


def next_trading_day(last_date: pd.Timestamp) -> pd.Series:
    """Return a 1-element Series with the next business day after last_date."""
    nxt = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=1)
    return pd.Series(nxt.tz_localize(None) if nxt.tz else nxt, name="timestamp")


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_predictor():
    """Load KronosPredictor (downloads weights from HuggingFace on first run)."""
    from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

    model_repo, tok_repo, max_ctx = _MODEL_REGISTRY[MODEL_KEY]
    print(f"[Kronos] Loading {MODEL_KEY} …", flush=True)
    tokenizer = KronosTokenizer.from_pretrained(tok_repo)
    model = Kronos.from_pretrained(model_repo)
    predictor = KronosPredictor(model, tokenizer, max_context=max_ctx)
    print(f"[Kronos] Model ready on {predictor.device}", flush=True)
    return predictor


# ── Prediction Logic ──────────────────────────────────────────────────────────

def run_predictions_direct(predictor, ticker_data: dict) -> dict:
    """
    Run N_RUNS stochastic passes for all tickers via predict_batch() and
    aggregate into win_probability / predicted_return / confidence.

    ticker_data: {ticker: (df, timestamps_series)}
    Returns: {ticker: {win_probability, predicted_return, confidence, direction}}
    """
    tickers = list(ticker_data.keys())
    dfs = [ticker_data[t][0] for t in tickers]
    timestamps = [ticker_data[t][1] for t in tickers]

    # Equalize lookback lengths across tickers (predict_batch requires uniform seq_len)
    min_len = min(len(df) for df in dfs)
    dfs = [df.iloc[-min_len:].reset_index(drop=True) for df in dfs]
    timestamps = [ts.iloc[-min_len:].reset_index(drop=True) for ts in timestamps]

    current_closes = np.array([df["close"].iloc[-1] for df in dfs])
    y_timestamps = [next_trading_day(ts.iloc[-1]) for ts in timestamps]

    # Accumulate predicted closes across N_RUNS stochastic passes
    all_pred_closes = {t: [] for t in tickers}

    for run_idx in range(N_RUNS):
        print(f"[Kronos] Run {run_idx + 1}/{N_RUNS} …", flush=True)
        try:
            pred_dfs = predictor.predict_batch(
                df_list=dfs,
                x_timestamp_list=timestamps,
                y_timestamp_list=y_timestamps,
                pred_len=1,
                T=TEMPERATURE,
                top_p=TOP_P,
                sample_count=1,
                verbose=False,
            )
            for i, ticker in enumerate(tickers):
                all_pred_closes[ticker].append(float(pred_dfs[i]["close"].iloc[0]))
        except Exception as e:
            print(f"[Kronos] WARNING: run {run_idx + 1} failed: {e}", flush=True)

    # Compute signal metrics
    results = {}
    for i, ticker in enumerate(tickers):
        pred_closes = np.array(all_pred_closes[ticker])
        if len(pred_closes) == 0:
            print(f"[Kronos] WARNING: all runs failed for {ticker}, skipping", flush=True)
            continue

        current_close = current_closes[i]
        pred_returns = (pred_closes - current_close) / current_close

        win_probability = float(np.mean(pred_returns > 0))
        predicted_return = float(np.mean(pred_returns))
        # Confidence: distance from 50/50 uncertainty, scaled to [0, 1]
        confidence = float(abs(win_probability - 0.5) * 2)

        if predicted_return > 0.005:
            direction = "bullish"
        elif predicted_return < -0.005:
            direction = "bearish"
        else:
            direction = "neutral"

        results[ticker] = {
            "win_probability": round(win_probability, 4),
            "predicted_return": round(predicted_return, 6),
            "confidence": round(confidence, 4),
            "direction": direction,
        }

    return results


def run_predictions_api(tickers: list) -> dict:
    """
    Call the Kronos REST API at API_URL.
    The API only returns a single mean prediction, so win_probability is derived
    from the sign and magnitude of the predicted return rather than sample voting.
    """
    try:
        import requests
    except ImportError:
        sys.exit("requests is required: pip install requests")

    import tempfile

    results = {}
    for ticker in tickers:
        try:
            df, _ = fetch_ohlcv(ticker, LOOKBACK)
        except Exception as e:
            print(f"[Kronos] WARNING: skipping {ticker}: {e}", flush=True)
            continue

        # Write a temp CSV for the API to read
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, dir=SIGNALS_DIR) as f:
            tmp_path = f.name
            df_csv = df.copy().reset_index()
            df_csv.columns = ["timestamps", "open", "high", "low", "close", "volume"]
            df_csv.to_csv(f, index=False)

        try:
            resp = requests.post(
                f"{API_URL}/api/predict",
                json={
                    "file_path": tmp_path,
                    "lookback": len(df),
                    "pred_len": 1,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "sample_count": N_RUNS,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("success"):
                raise RuntimeError(data.get("message", "unknown error"))

            pred_row = data["prediction_results"][-1]
            current_close = float(df["close"].iloc[-1])
            pred_close = float(pred_row["close"])
            pred_return = (pred_close - current_close) / current_close

            # API returns only the mean prediction — approximate win_prob from sign
            win_probability = 0.65 if pred_return > 0 else 0.35

            if pred_return > 0.005:
                direction = "bullish"
            elif pred_return < -0.005:
                direction = "bearish"
            else:
                direction = "neutral"

            # Confidence scales with how far the mean prediction is from zero
            confidence = round(min(abs(pred_return) * 20, 1.0), 4)

            results[ticker] = {
                "win_probability": round(win_probability, 4),
                "predicted_return": round(pred_return, 6),
                "confidence": confidence,
                "direction": direction,
            }
            print(f"[Kronos] {ticker}: {direction} ({pred_return:+.2%})", flush=True)

        except Exception as e:
            print(f"[Kronos] WARNING: API prediction failed for {ticker}: {e}", flush=True)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return results


# ── Output ────────────────────────────────────────────────────────────────────

def write_predictions_atomic(predictions: dict) -> None:
    """Write predictions to the signal bus using .tmp → rename for atomicity."""
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "predictions": predictions,
    }
    tmp_path = str(OUTPUT_FILE) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2)
    os.replace(tmp_path, str(OUTPUT_FILE))
    print(f"[Kronos] Wrote {len(predictions)} prediction(s) → {OUTPUT_FILE}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def load_tickers(args) -> list:
    """Resolve tickers from CLI args or a watchlist file."""
    if args.tickers:
        return [t.upper() for t in args.tickers]

    wl_path = Path(args.watchlist)
    if not wl_path.exists():
        sys.exit(f"Watchlist file not found: {wl_path}")

    tickers = []
    for line in wl_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tickers.extend(t.strip().upper() for t in line.split(",") if t.strip())
    if not tickers:
        sys.exit(f"No tickers found in {wl_path}")
    return tickers


def main():
    parser = argparse.ArgumentParser(
        description="Write Kronos day-ahead predictions to the shared signal bus"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Tickers to predict (e.g. --tickers MSFT AAPL TSLA)"
    )
    group.add_argument(
        "--watchlist", metavar="FILE",
        help="Path to watchlist file — one ticker per line, # for comments"
    )
    parser.add_argument(
        "--api", action="store_true",
        help=f"Use the Kronos REST API at {API_URL} instead of direct invocation"
    )
    args = parser.parse_args()

    tickers = load_tickers(args)
    print(f"[Kronos] Processing {len(tickers)} ticker(s): {', '.join(tickers)}", flush=True)

    if args.api:
        predictions = run_predictions_api(tickers)
    else:
        ticker_data = {}
        for ticker in tickers:
            try:
                df, ts = fetch_ohlcv(ticker, LOOKBACK)
                ticker_data[ticker] = (df, ts)
                print(f"[Kronos] {ticker}: {len(df)} bars, last close={df['close'].iloc[-1]:.4f}", flush=True)
            except Exception as e:
                print(f"[Kronos] WARNING: skipping {ticker}: {e}", flush=True)

        if not ticker_data:
            sys.exit("[Kronos] No valid ticker data fetched. Aborting.")

        predictor = load_predictor()
        predictions = run_predictions_direct(predictor, ticker_data)

    if not predictions:
        sys.exit("[Kronos] No predictions generated.")

    write_predictions_atomic(predictions)

    # Summary
    for ticker, p in predictions.items():
        print(
            f"  {ticker:6s}  {p['direction']:7s}  "
            f"return={p['predicted_return']:+.2%}  "
            f"win_prob={p['win_probability']:.0%}  "
            f"conf={p['confidence']:.0%}"
        )

    print("[Kronos] Done.", flush=True)


if __name__ == "__main__":
    main()
