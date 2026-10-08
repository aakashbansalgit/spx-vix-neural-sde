"""
Market data loading and smile extraction for SPX and VIX options.

Data is loaded once at module import from the data/misc/ directory and cached
in module-level globals (SPX, VIX, YIELD_CURVE, SPX_SPOT, VIX_SPOT).
The primary entry point is load_dataset().
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import torch
import datetime
from bs import implied_vol_np, callbs_np
from smile import Datetime

plt.style.use("classic")

_DATA_DIR = Path(os.environ.get("NSDE_DATA_DIR", Path(__file__).parent.parent / "data" / "misc"))
tau = datetime.timedelta(days=30)

# --- Module-level data cache ---
SPX         = pd.read_csv(_DATA_DIR / "spx_bid_ask.csv")
VIX         = pd.read_csv(_DATA_DIR / "vix_bid_ask.csv")
YIELD_CURVE = pd.read_csv(_DATA_DIR / "yield-curve.csv")
SPX_SPOT    = pd.read_csv(_DATA_DIR / "spx_spot.csv")
VIX_SPOT    = pd.read_csv(_DATA_DIR / "vix_spot.csv")

SPX_SPOT["date"] = pd.to_datetime(SPX_SPOT["date"], errors="coerce").dt.strftime("%Y-%m-%d")
VIX_SPOT["date"] = pd.to_datetime(VIX_SPOT["date"], errors="coerce").dt.strftime("%Y-%m-%d")

# Build a date-indexed rate lookup from the yield curve CSV
_yc = YIELD_CURVE.copy()
_yc["date"] = pd.to_datetime(_yc["date"])
treasury_rates = dict(zip(_yc["date"], _yc["rate"]))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def choose_fwd(asks_fwd, bids_fwd):
    """
    Choose the best forward price F such that it minimises |asks_fwd - F| + |F - bids_fwd|.
    Returns (fwd, fwd_bid, fwd_ask).
    """
    fwd0 = 0.5 * (bids_fwd[0] + asks_fwd[0])
    fwd = minimize(lambda f: (np.abs(asks_fwd - f) + np.abs(f - bids_fwd)).mean(), x0=fwd0).x[0]
    fwd_bid = bids_fwd[bids_fwd < fwd].max()
    fwd_ask = asks_fwd[asks_fwd > fwd].min()
    return fwd, fwd_bid, fwd_ask


def get_discount_factor(min_quote, max_quote, exdate):
    """
    Estimate the discount factor for an option expiring at exdate,
    using the nearest available Treasury rate.
    """
    min_quote = np.datetime64(pd.to_datetime(min_quote), "D")
    max_quote = np.datetime64(pd.to_datetime(max_quote), "D")
    exdate    = np.datetime64(pd.to_datetime(exdate), "D")

    available_dates = np.array(list(treasury_rates.keys()), dtype="datetime64[D]")
    available_rates = np.array(list(treasury_rates.values()))

    T_min = (exdate - min_quote).astype(int) / 365.0
    T_max = (exdate - max_quote).astype(int) / 365.0
    r_min = available_rates[np.argmin(np.abs(available_dates - min_quote))]
    r_max = available_rates[np.argmin(np.abs(available_dates - max_quote))]
    df = (np.exp(r_min * T_min) + np.exp(r_max * T_max)) / 2
    return df


def maturities_df(df):
    """Extract sorted Datetime objects from a raw options DataFrame."""
    T0 = df[["impl_volatility", "date"]].dropna().date.drop_duplicates().values
    assert len(T0) == 1, "Expected a single quote date in the slice."
    date_0 = str(T0[0])
    return [Datetime(d, T0=date_0) for d in df.exdate.drop_duplicates().values]


def compute_vol(fwd, maturity, strikes, prices, cp_flag):
    """Compute bid/ask implied vols for a set of strikes, filtering NaN results."""
    asks_vol = np.array([implied_vol_np(fwd, 0.0, maturity.t, k, c, cp_flag, nan=True)
                         for k, c in zip(strikes, prices["asks"])])
    bids_vol = np.array([implied_vol_np(fwd, 0.0, maturity.t, k, c, cp_flag, nan=True)
                         for k, c in zip(strikes, prices["bids"])])
    mask = np.logical_or(np.isnan(asks_vol), np.isnan(bids_vol))
    return strikes[~mask], asks_vol[~mask], bids_vol[~mask]


# ---------------------------------------------------------------------------
# Smile extraction
# ---------------------------------------------------------------------------

def _extract_smile(df, maturity):
    """
    Shared smile extraction logic for both SPX and VIX option chains.

    Computes the implied forward via put-call parity, applies a discount factor,
    then converts bid/ask prices to implied vols for calls and puts separately.
    Returns (smile dict, bid_ask dict).
    """
    calls = df.query("cp_flag == 'C'")
    puts  = df.query("cp_flag == 'P'")
    assert np.equal(calls.strike_price.values, puts.strike_price.values).all(), \
        "Call and put strikes must be aligned."

    strikes = calls.strike_price.values
    argsort = np.argsort(strikes)
    bid_ask = {
        "strikes": strikes[argsort],
        "calls": {"bids": calls.best_bid.values[argsort], "asks": calls.best_offer.values[argsort]},
        "puts":  {"bids": puts.best_bid.values[argsort],  "asks": puts.best_offer.values[argsort]},
    }

    ask_fwd = bid_ask["calls"]["asks"] - bid_ask["puts"]["bids"] + bid_ask["strikes"]
    bid_fwd = bid_ask["calls"]["bids"] - bid_ask["puts"]["asks"] + bid_ask["strikes"]
    fwd, fwd_bid, fwd_ask = choose_fwd(ask_fwd, bid_fwd)

    min_quote = pd.to_datetime(df["date"].min())
    max_quote = pd.to_datetime(df["date"].max())
    ex = pd.to_datetime(df["exdate"].iloc[0])
    df_factor = get_discount_factor(min_quote, max_quote, ex)

    fwd, fwd_bid, fwd_ask = fwd * df_factor, fwd_bid * df_factor, fwd_ask * df_factor

    strikes_c, vol_c_asks, vol_c_bids = compute_vol(fwd, maturity, bid_ask["strikes"], bid_ask["calls"], "C")
    strikes_p, vol_p_asks, vol_p_bids = compute_vol(fwd, maturity, bid_ask["strikes"], bid_ask["puts"],  "P")

    bid_ask["fwd"] = {"fwd": fwd, "bid": fwd_bid, "ask": fwd_ask, "asks": ask_fwd, "bids": bid_fwd}
    bid_ask["vol"] = {
        "c": {"strikes": strikes_c, "asks": vol_c_asks, "bids": vol_c_bids},
        "p": {"strikes": strikes_p, "asks": vol_p_asks, "bids": vol_p_bids},
    }

    smile_strikes = np.concatenate([strikes_p[strikes_p < fwd], strikes_c[strikes_c >= fwd]])
    smile_asks    = np.concatenate([vol_p_asks[strikes_p < fwd], vol_c_asks[strikes_c >= fwd]])
    smile_bids    = np.concatenate([vol_p_bids[strikes_p < fwd], vol_c_bids[strikes_c >= fwd]])

    smile = {
        "fwd": fwd, "fwd_bid": fwd_bid, "fwd_ask": fwd_ask,
        "strikes": smile_strikes,
        "asks": smile_asks, "bids": smile_bids,
        "mids": 0.5 * (smile_asks + smile_bids),
    }
    return smile, bid_ask


def extract_spx_smile(spx, maturity):
    """Extract the SPX volatility smile for a single maturity."""
    df = spx.query(f"exdate == '{maturity}'")
    return _extract_smile(df, maturity)


def extract_vix_smile(vix, maturity):
    """Extract the VIX volatility smile for a single maturity."""
    df = vix.query(f"exdate == '{maturity}'")
    return _extract_smile(df, maturity)


def extract_yield_curve(yield_curve):
    return {"days": yield_curve.days.values, "rate": yield_curve.rate.values}


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(date, maturities, full=False):
    """
    Load and return a dict containing SPX and VIX smiles, spot prices,
    and yield curve data for a given calibration date and set of maturities.

    Args:
        date       : calibration date string, e.g. '2021-10-01'
        maturities : dict with keys 'spx' and 'vix' listing expiry strings
        full       : if True, also store raw bid-ask data in the result

    Returns a dict with keys: vix_smiles, spx_smiles, vix_maturities,
        spx_maturities, yield_curve, spx_spot, vix_spot.
    """
    spx = SPX.loc[(SPX["date"] == date) & (SPX["exdate"].isin(maturities["spx"])), :]
    vix = VIX.loc[(VIX["date"] == date) & (VIX["exdate"].isin(maturities["vix"])), :]
    yield_curve = YIELD_CURVE.query(f"date == '{date}'")
    spx_spot    = SPX_SPOT.query(f"date == '{date}'")
    vix_spot    = VIX_SPOT.query(f"date == '{date}'")

    T0 = spx.date.drop_duplicates().values
    assert len(T0) == 1, f"Expected one quote date for {date}, found {T0}"

    vix_maturities = maturities_df(vix)
    spx_maturities = maturities_df(spx)
    yield_curve    = extract_yield_curve(yield_curve)

    data = {
        "vix_smiles": {}, "spx_smiles": {},
        "vix_maturities": vix_maturities, "spx_maturities": spx_maturities,
        "yield_curve": yield_curve, "spx_spot": spx_spot, "vix_spot": vix_spot,
    }

    for maturity in vix_maturities:
        smile, bid_ask = extract_vix_smile(vix, maturity)
        data["vix_smiles"][maturity] = {"smile": smile, "bid_ask": bid_ask if full else None}

    for maturity in spx_maturities:
        smile, bid_ask = extract_spx_smile(spx, maturity)
        data["spx_smiles"][maturity] = {"smile": smile, "bid_ask": bid_ask if full else None}

    return data


# ---------------------------------------------------------------------------
# Plotting (diagnostic only — requires full=True in load_dataset)
# ---------------------------------------------------------------------------

def plot(smiles, maturities):
    """Plot forward prices, implied vol bid-ask fills, and combined smiles for each maturity."""
    n = len(maturities)
    plt.figure(figsize=(27, 6 * n))
    for i, maturity in enumerate(maturities):
        bid_ask = smiles[maturity]["bid_ask"]
        smile   = smiles[maturity]["smile"]
        fwd     = bid_ask["fwd"]["fwd"]
        strikes = bid_ask["strikes"]

        plt.subplot(n, 3, i * 3 + 1)
        plt.scatter(strikes, bid_ask["fwd"]["asks"], color="blue",  label="fwd ask", s=5)
        plt.scatter(strikes, bid_ask["fwd"]["bids"], color="green", label="fwd bid", s=5)
        plt.axhline(fwd, color="black", label="chosen fwd")
        plt.xlabel("strike"); plt.ylabel("C - P + K"); plt.legend(); plt.title(maturity)

        plt.subplot(n, 3, i * 3 + 2)
        for flag, color in [("c", "blue"), ("p", "green")]:
            v = bid_ask["vol"][flag]
            plt.scatter(v["strikes"], v["asks"], color=color, s=1)
            plt.scatter(v["strikes"], v["bids"], color=color, s=1)
            plt.fill_between(v["strikes"], v["bids"], v["asks"], color=color, alpha=0.3)
        plt.axvline(fwd, color="black", label="fwd")
        plt.xlabel("strike"); plt.ylabel("implied vol."); plt.legend(); plt.title(maturity)

        plt.subplot(n, 3, i * 3 + 3)
        plt.scatter(smile["strikes"], smile["asks"], color="red", s=5)
        plt.scatter(smile["strikes"], smile["bids"], color="red", s=5)
        plt.fill_between(smile["strikes"], smile["bids"], smile["asks"],
                         color="red", alpha=0.3, label="bid-ask")
        plt.plot(smile["strikes"], smile["mids"], color="red", linestyle="--", label="mid")
        plt.axvline(fwd, color="black", label="fwd")
        plt.xlabel("strike"); plt.ylabel("implied vol."); plt.legend(); plt.title(maturity)
