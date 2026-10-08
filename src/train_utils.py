import torch
import time
import checkpoint
import torchsde
import json
import os
from smile import Datetime
from collections import OrderedDict
from vix import compute_VIX2
from run import run_model_ta_tb
from utils import (compute_smile_spx, compute_smile_vix,
                   compute_loss_smile_spx, compute_loss_smile_vix,
                   tensorize_smile, fwd_curve)
from config import DEVICE
from metrics import smile_metrics
from weights import dynamic_weights, BASE


def train(model, epoch, loss_func, checkpoint_func, save_step, max_epochs=None):
    """Main training loop. Runs until max_epochs, or until interrupted if None."""
    opt = model["optimizer"]
    last = None if max_epochs is None else epoch + max_epochs
    while last is None or epoch < last:
        opt.zero_grad(set_to_none=True)
        loss, output = loss_func(epoch)
        loss.backward()
        if epoch % save_step == 1 or (last is not None and epoch == last - 1):
            checkpoint_func(model, epoch, loss, output)
        opt.step()
        print(f"epoch {epoch:6d}  loss={loss.item():.6f}")
        epoch += 1
    return epoch


def wrap_loss_func(date, params, smiles, maturities, gen, dt, fSPX):
    """
    Build the loss function used during training.

    Creates a single BrownianInterval that is reused every epoch for variance reduction,
    then returns a closure that runs the SDE forward and computes the weighted sum of
    SPX, VIX forward, and VIX option losses.
    """
    batch_size = params["batch_size"]
    xy0 = torch.zeros([batch_size, 2], device=DEVICE)
    spx_maturities = maturities["spx"]
    vix_maturities = maturities["vix"]
    all_maturities = maturities["all"]

    spx_smiles = smiles["spx"]
    vix_smiles = smiles["vix"]

    T0 = Datetime(date)
    T_last = max(all_maturities)
    bm = torchsde.BrownianInterval(
        t0=T0.t, t1=T_last.t + 2 * dt, dt=dt,
        size=[batch_size, 2], device=DEVICE, entropy=42,
    )

    spot = fSPX(torch.tensor(0.0))
    t = [0] + [maturity.t for maturity in all_maturities]
    base = {"fVIX": params["w_fVIX"], "SPX": params["w_SPX"], "CVIX": params["w_CVIX"]}
    dynamic = params.get("weighting", "fixed") == "dynamic"
    sigma2 = params.get("sigma2", 1.0)
    # Base weights for the dynamic scheme are the ones the report settled on,
    # 20 / 3 / 2, rather than the published 30 / 3 / 2 used for fixed weights.
    dynamic_base = params.get("dynamic_base", BASE)
    coverage_buffer = params.get("coverage_buffer", 0.0)
    n_vix = len(vix_maturities)
    n_spx = len(spx_maturities)

    def loss_func(epoch, trajectories=False):
        XY, R = run_model_ta_tb(gen, spot, t, xy0, spx_smiles, bm)
        loss_dict = {"SPX": 0, "fVIX": 0, "CVIX": 0}
        model_smiles = {maturity.string: {} for maturity in maturities["spx_vix"]}

        for i, maturity in enumerate(all_maturities):
            if maturity in spx_maturities:
                XYi = XY[i + 1]
                smile_spx = compute_smile_spx(maturity, XYi, spot, spx_smiles)
                model_smiles[maturity]["spx"] = {"model": smile_spx, "market": spx_smiles[maturity]}
                use_w = maturity in spx_maturities[:2]
                loss_dict["SPX"] += compute_loss_smile_spx(smile_spx, spx_smiles[maturity], use_weights=use_w)

        for i, maturity in enumerate(all_maturities):
            if maturity in vix_maturities:
                XYi = XY[i + 1]
                j = all_maturities.index(maturity.plus_30d())
                Rij = R[j + 1] - R[i + 1]
                smile_vix = compute_smile_vix(maturity, XYi, Rij, vix_smiles)
                model_smiles[maturity]["vix"] = {"model": smile_vix, "market": vix_smiles[maturity]}
                loss_fVIX, loss_CVIX = compute_loss_smile_vix(smile_vix, vix_smiles[maturity], use_weights=False)
                loss_dict["fVIX"] += loss_fVIX
                loss_dict["CVIX"] += loss_CVIX

        parts = {"fVIX": loss_dict["fVIX"] / n_vix,
                 "SPX": loss_dict["SPX"] / n_spx,
                 "CVIX": loss_dict["CVIX"] / n_vix}
        w = dynamic_weights(parts, dynamic_base, sigma2) if dynamic else base
        loss = sum(w[k] * parts[k] for k in parts)

        output = {
            "loss": float(loss.detach().cpu()),
            "parts": {k: float(v.detach().cpu()) for k, v in parts.items()},
            "weights": w,
            "metrics": smile_metrics(model_smiles, coverage_buffer),
            "time": time.time(),
            "epoch": epoch,
            "smiles": model_smiles,
            "XY": XY if trajectories else None,
            "R": R if trajectories else None,
        }
        return loss, output

    return t, T0, loss_func


def prepare_data(data, date, maturities):
    """
    Convert raw market data into sorted Datetime-keyed smile dicts and build the forward curve.
    Filters each smile to a moneyness window around ATM:
        SPX : [0.15 * fwd, 1.25 * fwd]  (wide window to capture skew)
        VIX : [0.85 * fwd, 1.85 * fwd]
    """
    fSPX = fwd_curve(data)
    spx_maturities = sorted([Datetime(m, T0=date) for m in maturities["spx"]])
    vix_maturities = sorted([Datetime(m, T0=date) for m in maturities["vix"]])

    spx_smiles = OrderedDict({
        m: tensorize_smile(data["spx_smiles"], m, "SPX market", 40, 0.15, 1.25)
        for m in spx_maturities
    })
    vix_smiles = OrderedDict({
        m: tensorize_smile(data["vix_smiles"], m, "VIX market", 40, 0.85, 1.85)
        for m in vix_maturities
    })

    vix_maturities_plus30d = [m.plus_30d() for m in vix_maturities]
    spx_vix_maturities = sorted(set(spx_maturities + vix_maturities))
    all_maturities = sorted(set(spx_maturities + vix_maturities + vix_maturities_plus30d))

    smiles = {"vix": vix_smiles, "spx": spx_smiles}
    maturities = {
        "spx": spx_maturities, "vix": vix_maturities,
        "spx_vix": spx_vix_maturities, "all": all_maturities,
    }
    return smiles, maturities, fSPX


def checkpoint_func(model, epoch, loss, output):
    """Save a JSON snapshot of model smiles, a .ckpt of the model state, and
    one row of loss, weights, coverage and deviation to metrics.csv."""
    name = model["name"]
    js = json.dumps(output, indent=4, cls=checkpoint.Encoder)
    path = f"checkpoints/{name}/{epoch}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(js)
    checkpoint.save_checkpoint(epoch, loss, model)

    row = {"epoch": epoch, "time": output["time"], "loss": output["loss"]}
    row.update({f"loss_{k}": v for k, v in output["parts"].items()})
    row.update({f"w_{k}": v for k, v in output["weights"].items()})
    for key in ("spx_avg", "vix_avg"):
        if key in output["metrics"]:
            row[f"{key[:3]}_coverage"] = output["metrics"][key]["coverage"]
            row[f"{key[:3]}_deviation"] = output["metrics"][key]["deviation"]
    log = f"checkpoints/{name}/metrics.csv"
    new = not os.path.exists(log)
    with open(log, "a") as f:
        if new:
            f.write(",".join(row) + "\n")
        f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row.values()) + "\n")
    m = output["metrics"]
    print(f"  coverage spx {m.get('spx_avg', {}).get('coverage', float('nan')):.3f}"
          f"  vix {m.get('vix_avg', {}).get('coverage', float('nan')):.3f}"
          f"  deviation spx {m.get('spx_avg', {}).get('deviation', float('nan')):.3f}"
          f"  vix {m.get('vix_avg', {}).get('deviation', float('nan')):.3f}")
