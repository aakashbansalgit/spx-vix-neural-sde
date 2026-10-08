"""Unit tests for the metrics, the dynamic weights and warm starts.

Run from the repo root: python -m pytest tests  (or python tests/test_components.py)
"""
import math
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import coverage_rate, deviation_rate
from weights import BASE, dynamic_weights


def test_coverage_counts_strikes_inside_the_quote():
    bid = torch.tensor([0.18, 0.20, 0.22, 0.24])
    ask = torch.tensor([0.20, 0.22, 0.24, 0.26])
    model = torch.tensor([0.19, 0.25, 0.23, 0.30])       # in, out, in, out
    assert coverage_rate(model, bid, ask) == 0.5
    # widening each quote by 1.5 widths on each side brings 0.25 back inside
    # (the second quote now reaches 0.25) but not 0.30 (the last reaches 0.29)
    assert coverage_rate(model, bid, ask, buffer=1.5) == 0.75


def test_coverage_ignores_failed_implied_vols():
    bid, ask = torch.tensor([0.1, 0.1]), torch.tensor([0.2, 0.2])
    assert coverage_rate(torch.tensor([0.15, float("nan")]), bid, ask) == 1.0


def test_deviation_is_distance_from_mid_in_spreads():
    bid = torch.tensor([0.18, 0.20])
    ask = torch.tensor([0.20, 0.24])
    model = torch.tensor([0.19, 0.26])     # at mid; 0.04 above a 0.04-wide mid
    assert math.isclose(deviation_rate(model, bid, ask), 0.5, rel_tol=1e-6)


def test_dynamic_weights_lean_toward_the_worst_fitting_part():
    losses = {"fVIX": torch.tensor(0.1), "SPX": torch.tensor(0.8), "CVIX": torch.tensor(0.1)}
    w = dynamic_weights(losses, sigma2=1.0)
    assert all(v > 0 for v in w.values())
    # relative to its base, SPX gains the most because it holds most of the loss
    gain = {k: w[k] / BASE[k] for k in BASE}
    assert gain["SPX"] > gain["fVIX"] and math.isclose(gain["fVIX"], gain["CVIX"])
    assert math.isclose(gain["SPX"], math.exp(0.8))


def test_large_sigma2_recovers_the_fixed_weights():
    losses = {"fVIX": 1.0, "SPX": 5.0, "CVIX": 2.0}
    w = dynamic_weights(losses, sigma2=1e6)
    assert all(math.isclose(w[k], BASE[k], rel_tol=1e-5) for k in BASE)


def test_weights_carry_no_gradient():
    x = torch.tensor(0.5, requires_grad=True)
    w = dynamic_weights({"fVIX": x, "SPX": x * 2, "CVIX": x * 3})
    assert all(isinstance(v, float) for v in w.values())


def test_warm_start_copies_weights_and_resets_the_optimizer():
    import checkpoint
    net = lambda: torch.nn.Sequential(torch.nn.Linear(3, 4))
    src = {"nets": {"phi": net()}, "name": "src",
           "optimizer": None}
    src["optimizer"] = torch.optim.Adam(src["nets"]["phi"].parameters())
    path = Path(tempfile.mkdtemp()) / "src.ckpt"
    torch.save({"loss": 0.5, "epoch": 600, "phi": src["nets"]["phi"].state_dict(),
                "optimizer": src["optimizer"].state_dict()}, path)

    dst = {"nets": {"phi": net()}, "name": "dst", "optimizer": None}
    dst, epoch = checkpoint.warm_start(path, dst)
    assert epoch == 1
    for a, b in zip(src["nets"]["phi"].parameters(), dst["nets"]["phi"].parameters()):
        assert torch.equal(a, b)
    assert dst["optimizer"].state_dict()["state"] == {}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
