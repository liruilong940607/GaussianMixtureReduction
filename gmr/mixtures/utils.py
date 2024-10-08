import math
from typing import Tuple

import torch
from torch.distributions.utils import clamp_probs


def clamp_inf(val):
    b_max = torch.finfo(val.dtype).max
    b_min = torch.finfo(val.dtype).min
    return val.clamp(min=b_min, max=b_max)


def setdiff1d(tensor0: torch.Tensor, tensor1: torch.Tensor):
    """Find the set difference of two vector tensors.
    Return the unique values in tensor0 that are not in tensor1.
    It assumes each row of tensor0 contains corresponding row of tensor1

    Args:
        tensor0: tensor of (..., N)
        tensor1: tensor of (..., M) where N > M

    Returns:
        torch.Tensor: tensor of (..., N - M)

    """
    notin_idx = torch.ne(tensor0.unsqueeze(dim=-1), tensor1.unsqueeze(dim=-2)).all(
        dim=-1
    )
    num_remain = tensor0.size(-1) - tensor1.size(-1)
    out_tensor = torch.masked_select(tensor0, notin_idx).view(
        *tensor0.shape[:-1], num_remain
    )

    return out_tensor


def gauss_prob(
    x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, reduce=True
) -> torch.Tensor:
    """Return probability of x for (mu, var) distributed Gaussian

    Args:
        x: tensor of (..., D)
        mu: tensor of ([B], N, D)
        var: tensor of ([B], N, D, D)
        reduce: if True, do sum reduction on last dim

    Returns:
        torch.Tensor: tensor of (..., [B], N) if 'reduce' is True, (..., [B], N, D) otherwise

    """

    d = x.shape[-1]
    ex_x = x.view(x.shape[:-1] + (1,) * (mu.ndim - 1) + (x.shape[-1],))
    ex_var = torch.broadcast_to(var, x.shape[:-1] + var.shape)

    term0 = d * math.log(2 * math.pi) + torch.log(torch.linalg.det(var))
    term0 = term0 if reduce else term0[..., None]
    term1 = torch.einsum(
        "...i,...i->..." if reduce else "...i,...i->...i",
        ex_x - mu,
        torch.linalg.solve(ex_var, ex_x - mu),
    )
    prob = torch.exp(-0.5 * (term0 + term1))

    return prob


def integral_prod_gauss_prob(
    mu0: torch.Tensor,
    var0: torch.Tensor,
    mu1: torch.Tensor,
    var1: torch.Tensor,
    mode="self",
    same_dist=False,
) -> torch.Tensor:
    """Return integration of product of two gaussian

    Args:
        mu0: tensor of (..., N, D)
        var0: tensor of (..., N, D, D)
        mu1: tensor of (..., M, D)
        var1: tensor of (..., M, D, D)
        mode: 'self' or 'cross'
        same_dist: whether mu0 == mu1 and var0 == var1

    Returns:
        torch.Tensor: tensor of (..., N) if mode is 'self', (..., N, M) otherwise

    """

    d = mu0.size(-1)

    if mode == "self":
        diff_mu_ij = mu0 - mu1  # (..., N, D)
        sum_var_ij = var0 + var1  # (..., N, D, D)
    elif mode == "cross":
        diff_mu_ij = mu0.unsqueeze(dim=-2) - mu1.unsqueeze(dim=-3)  # (..., N, M, D)
        sum_var_ij = var0.unsqueeze(dim=-3) + var1.unsqueeze(
            dim=-4
        )  # (..., N, M, D, D)
    else:
        raise ValueError(f"mode(:{mode}) should be in ['self', 'cross'].")

    term0 = d * math.log(2 * math.pi) + torch.log(torch.linalg.det(sum_var_ij))
    if same_dist:
        term1 = 0.0
    else:
        term1 = torch.einsum(
            "...i,...i->...",
            diff_mu_ij,
            torch.linalg.solve(
                sum_var_ij + 1e-6 * torch.eye(d).to(sum_var_ij), diff_mu_ij
            ),
        )
    prob = clamp_probs(torch.exp(-0.5 * (term0 + term1)))

    return prob


def prod_gauss_dist(
    mu0: torch.Tensor,
    var0: torch.Tensor,
    mu1: torch.Tensor,
    var1: torch.Tensor,
    mode="self",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return gaussian parameters which is proportional to the product of two gaussian

    Args:
        mu0: tensor of (..., N, D)
        var0: tensor of (..., N, D, D)
        mu1: tensor of (..., M, D)
        var1: tensor of (..., M, D, D)
        mode: 'self' or 'cross'

    Returns:
        torch.Tensor: tensor of (..., N, D) if mode is 'self', (..., N, M, D) otherwise
        torch.Tensor: tensor of (..., N, D, D) if mode is 'self', (..., N, M, D, D) otherwise

    """

    d = mu0.size(-1)
    _inv_var0 = torch.linalg.inv(var0 + 1e-6 * torch.eye(d).to(var0))
    _inv_var1 = torch.linalg.inv(var1 + 1e-6 * torch.eye(d).to(var1))
    _inv_var_mu0 = torch.linalg.solve(var0 + 1e-6 * torch.eye(d).to(var0), mu0)
    _inv_var_mu1 = torch.linalg.solve(var1 + 1e-6 * torch.eye(d).to(var1), mu1)

    if mode == "self":
        _sum_inv_var = _inv_var0 + _inv_var1
        _sum_inv_var_mul_mu = _inv_var_mu0 + _inv_var_mu1
    elif mode == "cross":
        _sum_inv_var = _inv_var0.unsqueeze(dim=-3) + _inv_var1.unsqueeze(dim=-4)
        _sum_inv_var_mul_mu = _inv_var_mu0.unsqueeze(dim=-2) + _inv_var_mu1.unsqueeze(
            dim=-3
        )
    else:
        raise ValueError(f"mode(:{mode}) should be in ['self', 'cross'].")

    prod_var = torch.linalg.inv(_sum_inv_var)
    prod_mu = torch.einsum("...ij,...j->...i", prod_var, _sum_inv_var_mul_mu)

    return prod_mu, prod_var
