"""
Dynamic weights for the three parts of the calibration loss.

The published method fixes the weights on the SPX smile, the VIX future and
the VIX smile. Whichever part is largest then dominates the gradient and the
others stall. Here each weight is scaled up by an exponential of that part's
current share of the loss, so the weight moves toward whatever is fitting
worst:

    p_k = loss_k / total loss
    w_k = base_k * exp(p_k / sigma2)

The exponential keeps every weight positive and changes smoothly as the shares
move. sigma2 sets the sensitivity: smaller values react harder to imbalance,
larger values approach the fixed base weights. Shares are computed on detached
losses, so the weights steer the gradient without being differentiated
themselves.
"""
import math

BASE = {"fVIX": 20.0, "SPX": 3.0, "CVIX": 2.0}


def dynamic_weights(losses, base=BASE, sigma2=1.0):
    vals = {k: float(v.detach()) if hasattr(v, "detach") else float(v)
            for k, v in losses.items()}
    total = sum(vals.values())
    if total <= 0:
        return dict(base)
    return {k: base[k] * math.exp(vals[k] / total / sigma2) for k in base}
