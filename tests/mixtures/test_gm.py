from copy import deepcopy

import pytest
import numpy as np
from matplotlib import pyplot as plt

from mixtures.gm import calc_ise, kl_gm_comp, GM

gm = GM.sample_gm(
        n=3,
        d=2,
        pi_alpha=5*np.ones(3),
        mu_rng=[0., 3.],
        var_df=5,
        var_scale=1/5 * np.eye(2),
    )


@pytest.fixture(scope="session")
def plot(pytestconfig):
    return pytestconfig.getoption("plot")


def test_gm_mul(plot):
    gm0 = GM(pi=[0.3, 0.7], mu=np.array([1., 2.])[..., None], var=np.array([0.2, 0.1])[..., None, None])
    gm1 = GM(pi=[0.5, 0.5], mu=np.array([1.2, 2.7])[..., None], var=np.array([0.1, 0.3])[..., None, None])
    gm_prod = gm0 * gm1

    if plot:
        t = np.linspace(-0.5, 4.5, num=1000)[..., None]
        p0 = gm0.prob(t)
        p1 = gm1.prob(t)
        p_prod = gm_prod.prob(t)

        plt.plot(t, p0)
        plt.plot(t, p1)
        plt.plot(t, p_prod)
        plt.legend(['gm0', 'gm1', 'product'])
        plt.show()


def test_calc_ise():
    assert calc_ise(gm, gm) < 1e-9


def test_gm_prob(plot):
    t0, t1 = np.meshgrid(np.linspace(-1., 4., 100), np.linspace(-1., 4., 100))
    p = gm.prob(np.stack([t0, t1], axis=-1))

    if plot:
        print(gm)
        plt.contourf(t0, t1, p)
        plt.plot(gm.mu[:, 0], gm.mu[:, 1], 'x')
        plt.show()


def test_merge_gm(plot):
    cp_gm = deepcopy(gm)
    cp_gm.merge([[0, 1]])

    if plot:
        t0, t1 = np.meshgrid(np.linspace(-1., 4., 100), np.linspace(-1., 4., 100))
        p = cp_gm.prob(np.stack([t0, t1], axis=-1))
        print(cp_gm)
        plt.contourf(t0, t1, p)
        plt.plot(cp_gm.mu[:, 0], cp_gm.mu[:, 1], 'x')
        plt.show()


def test_kl_gm_comp():
    assert np.all(np.diag(kl_gm_comp(gm, gm)) < 1e-9)
