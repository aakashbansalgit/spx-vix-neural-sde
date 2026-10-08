import torch


def Monomial(XY, degree):
    """
    Build a monomial feature matrix from 2D state (X, Y).

    Includes:
        X^0 ... X^degree
        Y^1 ... Y^degree
        Cross terms X^k * Y^(degree+n-k) for n in [0, degree], k in [1, n-1]

    Returns a tensor of shape (N, n_features).
    """
    X, Y = XY[:, 0], XY[:, 1]
    P = []
    for k in range(degree + 1):
        P.append(torch.pow(X, k))
    for k in range(1, degree + 1):
        P.append(torch.pow(Y, k))
    for n in range(degree + 1):
        for k in range(1, n):
            P.append(P[k] * P[degree + n - k])
    return torch.stack(P, axis=-1)


def poly(XY, degree, k="M"):
    """Return a polynomial feature matrix. Currently only monomial basis ('M') is supported."""
    if k == "M":
        return Monomial(XY, degree)
    raise ValueError(f"Unknown polynomial kind '{k}'. Supported: 'M' (monomial).")
