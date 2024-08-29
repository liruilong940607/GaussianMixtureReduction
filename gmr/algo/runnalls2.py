from copy import deepcopy

import numpy as np
import torch
from gmr.mixtures.gm import GM


def fit_runnalls2(gm_ori: GM, L: int):
    """find gaussian mixture of L component which is close to original one by Runnalls' algorithm

    Args:
        gm_ori: target gaussian mixture
        L: the number of components of fitted mixture

    Returns:
        GM: fitted gaussian mixture

    """

    out_gm = deepcopy(gm_ori)

    while out_gm.n > L:
        c_ij = runnalls_cost(out_gm)

        indices = torch.argmin(c_ij, dim=0)
        matched = indices[indices] == torch.arange(len(indices), device=indices.device)

        # duplicated 2x
        ii = indices[matched].tolist()
        jj = indices[indices][matched].tolist()

        # select ii > jj to avoid duplication
        idx_list = []
        for i, j in zip(ii, jj):
            if i > j and len(idx_list) < out_gm.n - L:
                idx_list.append((i, j))

        # err = [c_ij[i, j] for i, j in idx_list]
        # print("ii", ii)
        # print("jj", jj)
        # print("idx_list", idx_list)
        # print("err", err)

        if len(idx_list) == 0:
            merge_idx = np.unravel_index(torch.argmin(c_ij).cpu(), c_ij.shape)
            idx_list = [merge_idx]
            # print("special case", idx_list)

        out_gm.merge(idx_list)
    # print(out_gm.n, out_gm.pi.shape, L)

    return out_gm


def runnalls_cost(gm: GM):
    """calculate the upper bound of KL divergence increase when two components are merged

    Returns:
        torch.Tensor: n by n matrix whose diagonal terms are infinite

    """

    pi_sum_ij = (gm.pi[..., None] + gm.pi)[..., None, None]
    pi_prod_ij = (gm.pi[..., None] * gm.pi)[..., None, None]
    term0 = gm.pi[..., None, None] * gm.var
    mu_diff = gm.mu.unsqueeze(dim=1) - gm.mu
    term1 = torch.einsum("...i,...j->...ij", mu_diff, mu_diff)
    var_ij = (
        1.0 / pi_sum_ij * (term0.unsqueeze(dim=1) + term0)
        + pi_prod_ij / pi_sum_ij**2 * term1
    )

    pi_det_ij = pi_sum_ij[..., 0, 0] * torch.log(torch.linalg.det(var_ij))
    pi_det_i = gm.pi * torch.log(torch.linalg.det(gm.var))
    c_ij = 0.5 * (pi_det_ij - pi_det_i[..., None] - pi_det_i)

    # make diagonal term infinite
    c_ij[torch.arange(gm.n), torch.arange(gm.n)] = torch.inf * torch.ones(gm.n).type_as(
        c_ij
    )

    return c_ij
