import numpy as np

import matplotlib.pyplot as plt

from cowa import fit_cowa
from gm import sample_gm, gm_prob, calc_ise, GM
from gmrc import fit_gmrc
from runnalls import fit_runnalls
from west import fit_west



def main():
    n = 10
    pi = np.array([0.03, 0.18, 0.12, 0.19, 0.02, 0.16, 0.06, 0.1, 0.08, 0.06])
    mu = np.array([1.45, 2.2, 0.67, 0.48, 1.49, 0.91, 1.01, 1.42, 2.77, 0.89])
    var = np.array([0.0487, 0.0305, 0.1171, 0.0174, 0.0295, 0.0102, 0.0323, 0.038, 0.0115, 0.0679])
    gm = GM(n=n, pi=pi, mu=mu, var=var)

    runnalls_gm = fit_runnalls(gm, L=5)
    west_gm = fit_west(gm, L=5)
    gmrc_gm = fit_gmrc(gm, L=5)
    cowa_gm = fit_cowa(gm, L=5)

    t = np.linspace(-1, 4, num=1000)
    p = gm_prob(t, gm)
    runnalls_p = gm_prob(t, runnalls_gm)
    west_p = gm_prob(t, west_gm)
    gmrc_p = gm_prob(t, gmrc_gm)
    cowa_p = gm_prob(t, cowa_gm)

    plt.plot(t, p, '--', c='k')
    plt.plot(t, runnalls_p)
    plt.plot(t, west_p)
    plt.plot(t, gmrc_p)
    plt.plot(t, cowa_p)
    plt.legend(['full', 'Runnalls', 'West', 'GMRC', 'COWA'])
    plt.show()

    ise_runnalls = calc_ise(gm, runnalls_gm)
    ise_west = calc_ise(gm, west_gm)
    ise_gmrc = calc_ise(gm, gmrc_gm)
    ise_cowa = calc_ise(gm, cowa_gm)

    print(f"ISE Runnalls: {ise_runnalls:.5f}")
    print(f"ISE West: {ise_west:.5f}")
    print(f"ISE GMRC: {ise_gmrc:.5f}")
    print(f"ISE COWA: {ise_cowa:.5f}")


if __name__ == '__main__':
    main()
