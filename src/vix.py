import torch
from polynomial import poly
from config import DEVICE


def _tXYfy(t, XY):
    """Broadcast scalar time t across all paths and concatenate with state XY."""
    N = XY.shape[1]
    t = t[:, None, None] * torch.ones([t.shape[0], N, 1])
    return torch.cat([t.to(DEVICE), XY.to(DEVICE)], axis=-1)


def _do_lstsq(A, R):
    """Solve the least-squares system A @ x = R via QR decomposition."""
    q, r = torch.linalg.qr(A)
    return torch.linalg.solve_triangular(r, q.T @ R, upper=True)


def compute_VIX2(XY12, R, degree, k="M"):
    """
    Estimate VIX² via polynomial regression of realized variance onto state (X, Y).

    VIX² ≈ E[R | X, Y] = PSI(X, Y) @ coefficients
    where R = (2/τ) * ∫ σ_X² dt is the 30-day annualized realized variance.

    Only paths with a positive fitted VIX² are returned (negatives are filtered out).
    """
    if XY12.ndim == 3:
        XY12 = XY12[0]
    elif XY12.ndim != 2:
        raise ValueError(f"XY12 must be 2D or 3D, got shape {XY12.shape}")
    PSI = poly(XY12, degree, k)
    coeffs = _do_lstsq(PSI, R)
    VIX2 = PSI @ coeffs
    VIX2 = VIX2[VIX2 > 0]
    return VIX2.unsqueeze(-1)


def compute_VIX2_nested(gen12, T1, T2, TAU, XY1, sub_batch_size, batch_size_nested, k):
    """
    Nested Monte Carlo estimate of VIX² at time T1 by averaging over inner paths from T1 → T2.
    Used for high-quality inference (not training).
    """
    R_nested = torch.zeros([k, sub_batch_size, 1], device=DEVICE)
    n = batch_size_nested // k
    XY1_nested = XY1[:sub_batch_size].repeat(k, 1)
    with torch.no_grad():
        for j in range(n):
            _, _, R, _ = gen12([T1.t, T2.t], XY1_nested, bm=None)
            RR = (2 * R[-1] / TAU).reshape([k, sub_batch_size, 1])
            R_nested += RR
            print(f"{j + 1}/{n}", end="\r")
        R_nested /= n
    return R_nested.mean(axis=0)
