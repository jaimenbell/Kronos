"""
Kronos prediction REST API.

Thin Flask wrapper around KronosPredictor that fetches live OHLCV data via
yfinance and returns a standardized prediction response compatible with the
whale-tracker KronosClient.

Endpoints:
    POST /predict   {"ticker": "AAPL", "horizon_days": 5}
    GET  /health

Response (POST /predict):
    {
        "ticker": "AAPL",
        "horizon_days": 5,
        "predicted_return": 0.072,   # e.g. +7.2% over horizon
        "win_probability": 0.64,     # fraction of predicted candles above current close
        "confidence": 0.81           # path-consistency in the predicted direction [0.3, 0.95]
    }

Run:
    python kronos_api.py                         # Kronos-small, auto device, port 8000
    python kronos_api.py --model kronos-mini     # Faster, lighter model
    python kronos_api.py --device cpu --port 9000

Extra dependencies (beyond Kronos requirements.txt):
    pip install flask flask-cors yfinance
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS

# Add project root so `from model import ...` works from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Kronos, KronosTokenizer, KronosPredictor

app = Flask(__name__)
CORS(app)

# ── Model registry ─────────────────────────────────────────────────────────────
_MODEL_CONFIGS: dict[str, tuple[str, str, int]] = {
    # key: (model_id, tokenizer_id, max_context)
    "kronos-mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-2k",  2048),
    "kronos-small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "kronos-base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base", 512),
}

# Loaded once at startup; None until load_model() is called
_predictor: Optional[KronosPredictor] = None
_model_name: str = ""

# ── Prediction constants ───────────────────────────────────────────────────────
_LOOKBACK = 400   # Historical daily candles fed to the model
_TEMPERATURE = 1.0
_TOP_P = 0.9
_SAMPLE_COUNT = 1  # Single stochastic path (fast). Increase for smoother avg.


# ── Data fetching ──────────────────────────────────────────────────────────────

def _fetch_daily_ohlcv(ticker: str, lookback: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Download daily OHLCV from Yahoo Finance and return (x_df, x_timestamp).

    x_df has columns: open, high, low, close, volume  (last `lookback` rows)
    x_timestamp is a pd.Series of datetime values aligned to x_df.

    Raises ValueError on empty / insufficient data.
    """
    # Download ~3 years; yfinance strips non-trading days so we need the buffer
    raw = yf.download(ticker, period="3y", progress=False, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"No Yahoo Finance data for ticker {ticker!r}")

    # yfinance ≥ 0.2.x may return MultiIndex columns for single-ticker downloads
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.dropna()

    if len(df) < lookback:
        raise ValueError(
            f"Ticker {ticker!r}: only {len(df)} trading days available, "
            f"need at least {lookback}"
        )

    df = df.iloc[-lookback:].copy()
    # DatetimeIndex → Series so Kronos's calc_time_stamps doesn't get confused
    x_timestamp = pd.Series(df.index, name="timestamps")
    df = df.reset_index(drop=True)  # drop DatetimeIndex so iloc stays clean

    return df, x_timestamp


def _future_timestamps(last_date: pd.Timestamp, n_days: int) -> pd.Series:
    """Return n_days business-day timestamps starting after last_date."""
    return pd.Series(
        pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days),
        name="timestamps",
    )


# ── Metric derivation ─────────────────────────────────────────────────────────

def _derive_metrics(
    pred_df: pd.DataFrame,
    current_close: float,
) -> tuple[float, float, float]:
    """
    Derive the three whale-tracker metrics from a Kronos prediction path.

    predicted_return:
        (final predicted close - current close) / current close

    win_probability:
        Fraction of predicted candles whose close is above current close.
        Proxies P(price > entry) over the horizon.

    confidence:
        How consistently the predicted path moves in the direction of the
        final predicted return.  Maps path-consistency [0, 1] → [0.30, 0.95].
        Higher = model "commits" to the direction throughout the horizon.
    """
    closes = pred_df["close"].values
    final_close = float(closes[-1])
    predicted_return = (final_close - current_close) / current_close

    win_probability = float(np.sum(closes > current_close) / len(closes))

    # Directional consistency over the whole path
    if predicted_return >= 0:
        consistency = float(np.sum(closes >= current_close) / len(closes))
    else:
        consistency = float(np.sum(closes <= current_close) / len(closes))

    confidence = 0.30 + consistency * 0.65   # range [0.30, 0.95]

    return predicted_return, win_probability, confidence


# ── Flask routes ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick liveness + readiness check."""
    loaded = _predictor is not None
    device = str(_predictor.device) if loaded else None
    return jsonify({
        "status": "ok" if loaded else "model_not_loaded",
        "model": _model_name or None,
        "device": device,
        "loaded": loaded,
    })


@app.post("/predict")
def predict():
    """
    Generate a price prediction for a ticker.

    Request body (JSON):
        ticker        str   required — e.g. "AAPL"
        horizon_days  int   optional — forecast horizon in trading days (default 5)

    Response (JSON):
        ticker            str
        horizon_days      int
        predicted_return  float   e.g. 0.072 = +7.2%
        win_probability   float   e.g. 0.64
        confidence        float   e.g. 0.81
    """
    if _predictor is None:
        return jsonify({"error": "Model not loaded — server still initializing"}), 503

    body = request.get_json(silent=True) or {}
    ticker = (body.get("ticker") or "").upper().strip()
    try:
        horizon_days = int(body.get("horizon_days", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "horizon_days must be an integer"}), 400

    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    if not (1 <= horizon_days <= 30):
        return jsonify({"error": "horizon_days must be between 1 and 30"}), 400

    try:
        # 1. Fetch historical OHLCV
        x_df, x_timestamp = _fetch_daily_ohlcv(ticker, _LOOKBACK)
        current_close = float(x_df["close"].iloc[-1])
        last_date = pd.Timestamp(x_timestamp.iloc[-1])

        # 2. Build future timestamps for the prediction window
        y_timestamp = _future_timestamps(last_date, horizon_days)

        # 3. Run Kronos prediction
        pred_df = _predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=horizon_days,
            T=_TEMPERATURE,
            top_p=_TOP_P,
            sample_count=_SAMPLE_COUNT,
            verbose=False,
        )

        # 4. Derive the three metrics whale-tracker expects
        predicted_return, win_probability, confidence = _derive_metrics(
            pred_df, current_close
        )

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify({
        "ticker": ticker,
        "horizon_days": horizon_days,
        "predicted_return": round(predicted_return, 6),
        "win_probability": round(win_probability, 4),
        "confidence": round(confidence, 4),
    })


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_key: str = "kronos-small", device: Optional[str] = None) -> None:
    """
    Load Kronos model weights into the global predictor.  Called once at startup.

    Args:
        model_key: One of "kronos-mini", "kronos-small", "kronos-base".
        device:    "cpu", "cuda:0", "mps", or None to auto-detect.
    """
    global _predictor, _model_name

    if model_key not in _MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model {model_key!r}. "
            f"Choose from: {list(_MODEL_CONFIGS)}"
        )

    model_id, tokenizer_id, max_context = _MODEL_CONFIGS[model_key]

    print(f"[kronos_api] Loading tokenizer  : {tokenizer_id}")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)

    print(f"[kronos_api] Loading model      : {model_id}")
    model = Kronos.from_pretrained(model_id)

    print(f"[kronos_api] Building predictor : device={device or 'auto'}, max_context={max_context}")
    _predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
    _model_name = model_key

    print(f"[kronos_api] Ready — {model_key} on {_predictor.device}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos prediction REST API")
    parser.add_argument(
        "--model", default="kronos-small",
        choices=list(_MODEL_CONFIGS),
        help="Model variant to load (default: kronos-small)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Inference device: cpu | cuda:0 | mps (default: auto-detect)",
    )
    parser.add_argument(
        "--port", default=8000, type=int,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    load_model(model_key=args.model, device=args.device)
    print(f"[kronos_api] Listening on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
