import pytest
import numpy as np
from matplotlib import pyplot as plt

from algo.brute_force import fit_brute_force
from mixtures.gm import GM

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


def test_brute_force(plot):
    m_gm = fit_brute_force(gm, L=2)

    if plot:
        t0, t1 = np.meshgrid(np.linspace(-1., 4., 100), np.linspace(-1., 4., 100))
        m_p = m_gm.prob(np.stack([t0, t1], axis=-1))
        print(m_gm)
        plt.contourf(t0, t1, m_p)
        plt.plot(m_gm.mu[:, 0], m_gm.mu[:, 1], 'x')
        plt.show()
