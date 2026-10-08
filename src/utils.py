import torch
import numpy as np
from smile import Smile
from bs import implied_vol
from scipy.interpolate import interp1d
from vix import compute_VIX2
from config import DEVICE


def compute_smile_spx(maturity, xy, spot, spx_smiles):
    """Compute model-implied SPX volatility smile from Monte Carlo paths."""
    strikes = spx_smiles[maturity].strikes
    t0 = torch.tensor(0.0)
    s = spot * xy[:, :1].exp()
    f = s.mean().detach()
    c = (s - strikes).relu().mean(axis=0)
    p = (strikes - s).relu().mean(axis=0)
    t = torch.tensor(maturity.t, device=DEVICE)
    i = implied_vol(f, t0, t, strikes, c, p).to(DEVICE)
    return Smile(
        "SPX model", maturity,
        {"strikes": strikes, "fwd": f, "fwd_ask": f, "fwd_bid": f,
         "mids": i, "asks": i, "bids": i, "calls": c, "puts": p},
    )


def compute_smile_vix(maturity, XY, R, vix_smiles):
    """Compute model-implied VIX volatility smile from Monte Carlo paths."""
    VIX2 = compute_VIX2(XY, R, degree=8, k="M")
    strikes = vix_smiles[maturity].strikes
    VIX = 100 * VIX2.sqrt()
    fVIX = VIX.mean()
    t0 = torch.tensor(0.0)
    CVIX = (VIX - strikes).relu().mean(axis=0)
    PVIX = (strikes - VIX).relu().mean(axis=0)
    t = torch.tensor(maturity.t, device=DEVICE)
    IVIX = implied_vol(fVIX, t0, t, strikes, CVIX, PVIX).to(DEVICE)
    return Smile(
        "VIX model", maturity,
        {"strikes": strikes, "fwd": fVIX, "fwd_bid": fVIX, "fwd_ask": fVIX,
         "mids": IVIX, "asks": IVIX, "bids": IVIX, "calls": CVIX, "puts": PVIX},
    )


def compute_loss_smile_spx(smile1, smile2, use_weights=False):
    """
    MSE loss between model and market SPX smiles (relative implied vol differences).
    Optionally weights strikes by inverse bid-ask spread (tighter = more trusted).
    """
    weights = 1 / (smile2.asks - smile2.bids) if use_weights else torch.ones_like(smile2.mids)
    loss = (weights * ((smile1.mids / smile2.mids - 1) ** 2)).sum() / weights.sum()
    return loss


def compute_loss_smile_vix(smile1, smile2, use_weights=False):
    """
    Loss between model and market VIX smiles. Returns (forward loss, options loss) separately
    so the training loop can apply different weights to each component.
    """
    loss_future = (smile1.fwd / smile2.fwd - 1) ** 2
    strikes = smile1.strikes
    weights = 1 / (smile2.asks - smile2.bids) if use_weights else torch.ones_like(smile2.mids)
    loss_calls = weights * ((smile1.calls / smile2.calls - 1) ** 2)
    loss_puts = weights * ((smile1.puts / smile2.puts - 1) ** 2)
    mask_calls = strikes > smile1.fwd
    mask_puts = strikes <= smile1.fwd
    loss_options = (loss_calls[mask_calls].sum() + loss_puts[mask_puts].sum()) / weights.sum()
    return loss_future, loss_options


def tensorize_smile(data, maturity, instrument, nb_points, alpha1=None, alpha2=None):
    """
    Convert a raw smile dict to a Smile tensor object on DEVICE.
    Optionally filters to strikes in [alpha1 * fwd, alpha2 * fwd] and
    downsamples to approximately nb_points strikes.
    """
    raw = data[maturity]["smile"]
    ts = lambda arr: torch.tensor(arr, device=DEVICE, dtype=torch.float32)
    fwd, fwd_bid, fwd_ask = ts(raw["fwd"]), ts(raw["fwd_bid"]), ts(raw["fwd_ask"])
    strikes, mids, asks, bids = ts(raw["strikes"]), ts(raw["mids"]), ts(raw["asks"]), ts(raw["bids"])

    if alpha1 and alpha2:
        mask = (alpha1 * fwd <= strikes) & (strikes <= alpha2 * fwd)
        strikes, mids, asks, bids = strikes[mask], mids[mask], asks[mask], bids[mask]

    k = max(1, strikes.shape[0] // nb_points + 1)
    strikes, mids, asks, bids = strikes[::k], mids[::k], asks[::k], bids[::k]
    return Smile(instrument, maturity,
                 {"fwd": fwd, "fwd_bid": fwd_bid, "fwd_ask": fwd_ask,
                  "strikes": strikes, "mids": mids, "bids": bids, "asks": asks,
                  "calls": None, "puts": None})


def fwd_curve(data):
    """
    Build a callable forward curve F(t) by linear interpolation of
    spot + SPX futures prices across maturities.
    """
    spx_maturities = data["spx_maturities"]
    t = [0.0] + [m.t for m in spx_maturities]
    fwd = (data["spx_spot"].spot.values.tolist()
           + [data["spx_smiles"][m]["smile"]["fwd"] for m in spx_maturities])
    fSPX_func = interp1d(t, fwd, kind="linear", fill_value="extrapolate")
    return lambda t: float(fSPX_func(np.maximum(t if isinstance(t, (int, float)) else t.cpu(), 0.0)))
