"""
Writes a synthetic option market in the same CSV layout as the real data, so
the full training pipeline can run without proprietary quotes.

SPX quotes come from a Black smile with a downside skew, VIX quotes from a
Black smile with an upside skew, both at a zero rate with bid-ask spreads
proportional to price. The two surfaces are not jointly arbitrage-free, so
the model will not fit them perfectly. That is fine: the point is that every
stage, from smile extraction through training, checkpointing and the
coverage and deviation metrics, runs on input in the real format.

    python tests/make_synthetic_market.py out_dir
"""
import json
import sys
from math import log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

DATE = "2021-10-01"
SPX_SPOT, VIX_SPOT = 4357.0, 21.0
MATURITIES = json.loads((Path(__file__).parents[1] / "src" / "maturities_monthly.json").read_text())


def black(F, K, T, vol, call):
    s = vol * sqrt(T)
    d1 = (log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    return F * norm.cdf(d1) - K * norm.cdf(d2) if call else K * norm.cdf(-d2) - F * norm.cdf(-d1)


def chain(F, strikes, T, vol_fn, min_tick):
    rows = []
    for K in strikes:
        vol = vol_fn(log(K / F))
        for cp in ("C", "P"):
            mid = black(F, K, T, vol, cp == "C")
            half = max(min_tick, 0.02 * mid)
            rows.append(dict(cp_flag=cp, strike_price=K, impl_volatility=vol,
                             best_bid=max(mid - half, min_tick / 2), best_offer=mid + half))
    return rows


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = pd.Timestamp(DATE)

    spx = []
    for exp in MATURITIES["spx"]:
        T = (pd.Timestamp(exp) - t0).days / 365
        strikes = np.arange(0.6 * SPX_SPOT, 1.4 * SPX_SPOT, 25.0).round()
        for r in chain(SPX_SPOT, strikes, T, lambda k: float(np.clip(0.17 - 0.6 * k, 0.08, 0.7)), 0.05):
            spx.append(dict(date=DATE, exdate=exp, **r))
    vix = []
    for i, exp in enumerate(MATURITIES["vix"]):
        T = (pd.Timestamp(exp) - t0).days / 365
        F = VIX_SPOT + 0.8 * (i + 1)          # contango
        for r in chain(F, np.arange(10.0, 60.0, 1.0), T, lambda k: float(np.clip(0.95 + 0.5 * k, 0.4, 2.5)), 0.01):
            vix.append(dict(date=DATE, exdate=exp, **r))

    pd.DataFrame(spx).to_csv(out / "spx_bid_ask.csv", index=False)
    pd.DataFrame(vix).to_csv(out / "vix_bid_ask.csv", index=False)
    pd.DataFrame(dict(date=[DATE] * 3, days=[30, 90, 365], rate=[0.0] * 3)).to_csv(out / "yield-curve.csv", index=False)
    pd.DataFrame(dict(date=[DATE], spot=[SPX_SPOT])).to_csv(out / "spx_spot.csv", index=False)
    pd.DataFrame(dict(date=[DATE], spot=[VIX_SPOT])).to_csv(out / "vix_spot.csv", index=False)
    print(f"wrote {len(spx)} SPX and {len(vix)} VIX quotes to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
