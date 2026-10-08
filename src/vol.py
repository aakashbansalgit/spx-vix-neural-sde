import torch
from config import DEVICE

Rho_min = torch.tensor(-0.99999, device=DEVICE)
Rho_max = torch.tensor(+0.99999, device=DEVICE)


def V_and_MuY_rho_tanh(model, alpha=1):
    """
    Parameterises the diffusion matrix and VIX-squared drift using tanh activations.

    Outputs:
        SigmaX = 1 + tanh(U[0])         in (0, 2)
        SigmaY = alpha * (1 + tanh(U[1])) in (0, 2*alpha)
        Rho    = tanh(U[2])              in (-1, 1)
        MuY    = U[3]                    unbounded
    """
    def V_and_MuY(tXY):
        batch_size = tXY.shape[0]
        zeros = torch.zeros([batch_size, 1], device=DEVICE)
        U = model["nets"]["phi"](tXY)
        SigmaX = 1 + torch.tanh(U[:, [0]])
        SigmaY = alpha * (1 + torch.tanh(U[:, [1]]))
        Rho = torch.tanh(U[:, [2]])
        Rho = torch.clamp(Rho, Rho_min, Rho_max)
        RhoP = torch.sqrt(1 - Rho ** 2)
        MuY = U[:, [3]]
        V = torch.concat([SigmaX, zeros, Rho * SigmaY, RhoP * SigmaY], axis=-1)
        return V, MuY

    return V_and_MuY


def V_and_MuY_rho_exp(model):
    """
    Alternative parameterisation using exponential activations for the volatilities.
    SigmaX = exp(U[0]), SigmaY = exp(U[1]) — unbounded above zero.
    """
    def V_and_MuY(tXY):
        batch_size = tXY.shape[0]
        zeros = torch.zeros([batch_size, 1], device=DEVICE)
        U = model["nets"]["phi"](tXY)
        SigmaX = torch.exp(U[:, [0]])
        SigmaY = torch.exp(U[:, [1]])
        Rho = torch.tanh(U[:, [2]])
        Rho = torch.clamp(Rho, Rho_min, Rho_max)
        RhoP = torch.sqrt(1 - Rho ** 2)
        MuY = U[:, [3]]
        V = torch.concat([SigmaX, zeros, Rho * SigmaY, RhoP * SigmaY], axis=-1)
        return V, MuY

    return V_and_MuY


def V_and_MuY_rho_softplus(model):
    """
    Alternative parameterisation using softplus activations for the volatilities.
    """
    def V_and_MuY(tXY):
        batch_size = tXY.shape[0]
        zeros = torch.zeros([batch_size, 1], device=DEVICE)
        U = model["nets"]["phi"](tXY)
        SigmaX = torch.nn.functional.softplus(U[:, [0]])
        SigmaY = torch.nn.functional.softplus(U[:, [1]])
        Rho = torch.tanh(U[:, [2]])
        Rho = torch.clamp(Rho, Rho_min, Rho_max)
        RhoP = torch.sqrt(1 - Rho ** 2)
        MuY = U[:, [3]]
        V = torch.concat([SigmaX, zeros, Rho * SigmaY, RhoP * SigmaY], axis=-1)
        return V, MuY

    return V_and_MuY
