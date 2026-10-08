"""
Fit diagnostics measured against the market's bid-ask, not against the loss.

Training loss is a relative squared error averaged over strikes and
instruments, which says little about whether a trader could quote off the
model. These two numbers do:

    coverage rate  = share of strikes where model IV lies in [bid IV, ask IV]
    deviation rate = mean over strikes of |model IV - mid IV| / (ask IV - bid IV)

Coverage is what a desk asks about. It is a step function of the fit, so it
is noisy on sparse strikes, and an optional buffer widens each quote by a
fraction of its own width to smooth it. Deviation is continuous, which makes
it the better curve to watch while training runs. A deviation below 0.5 means
the model sits inside the bid-ask on average.
"""
import torch


def coverage_rate(model_iv, bid_iv, ask_iv, buffer=0.0):
    width = ask_iv - bid_iv
    lo, hi = bid_iv - buffer * width, ask_iv + buffer * width
    ok = torch.isfinite(model_iv)
    inside = (model_iv >= lo) & (model_iv <= hi) & ok
    return float(inside.sum() / ok.sum()) if ok.any() else float("nan")


def deviation_rate(model_iv, bid_iv, ask_iv):
    mid = 0.5 * (bid_iv + ask_iv)
    width = ask_iv - bid_iv
    ok = torch.isfinite(model_iv) & (width > 0)
    return float(((model_iv[ok] - mid[ok]).abs() / width[ok]).mean()) if ok.any() else float("nan")


def smile_metrics(model_smiles, buffer=0.0):
    """Coverage and deviation for every maturity and instrument, plus the
    averages across maturities. `model_smiles` is the dict built in the loss
    function: {maturity: {"spx" | "vix": {"model": Smile, "market": Smile}}}."""
    rows, totals = {}, {}
    for maturity, pair in model_smiles.items():
        for inst, sm in pair.items():
            m, mk = sm["model"].mids.detach().cpu(), sm["market"]
            bid, ask = mk.bids.detach().cpu(), mk.asks.detach().cpu()
            cov = coverage_rate(m, bid, ask, buffer)
            dev = deviation_rate(m, bid, ask)
            rows[f"{inst}_{maturity}"] = {"coverage": cov, "deviation": dev}
            totals.setdefault(inst, []).append((cov, dev))
    for inst, vals in totals.items():
        rows[f"{inst}_avg"] = {
            "coverage": sum(v[0] for v in vals) / len(vals),
            "deviation": sum(v[1] for v in vals) / len(vals),
        }
    return rows
