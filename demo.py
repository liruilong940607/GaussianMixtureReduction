import time

import matplotlib.pyplot as plt
import torch
from gmr import algo, mixtures

preset_gm = False


def main():

    if preset_gm:
        pi = torch.tensor([0.03, 0.18, 0.12, 0.19, 0.02, 0.16, 0.06, 0.1, 0.08, 0.06])
        mu = torch.tensor([1.45, 2.2, 0.67, 0.48, 1.49, 0.91, 1.01, 1.42, 2.77, 0.89])[
            ..., None
        ]
        var = torch.tensor(
            [
                0.0487,
                0.0305,
                0.1171,
                0.0174,
                0.0295,
                0.0102,
                0.0323,
                0.038,
                0.0115,
                0.0679,
            ]
        )[..., None, None]
        gm = mixtures.GM(pi=pi, mu=mu, var=var)
    else:
        gm = mixtures.GM.sample_gm(
            n=1000,
            d=1,
            pi_alpha=torch.ones(1000),
            mu_rng=[0.0, 3.0],
            var_df=torch.tensor(2.0),
            var_scale=torch.eye(1) * 0.1,
        )

    # reduce gaussian mixtures
    tic = time.time()
    runnalls_gm = algo.fit_runnalls(gm, L=5)
    print(f"Runnalls took {time.time() - tic:.5f} seconds")
    tic = time.time()
    # west_gm = algo.fit_west(gm, L=5)
    # print(f"West took {time.time() - tic:.5f} seconds")
    # tic = time.time()
    runnalls2_gm = algo.fit_runnalls2(gm, L=5)
    print(f"Runnalls2 took {time.time() - tic:.5f} seconds")
    tic = time.time()
    # cowa_gm = algo.fit_cowa(gm, L=5)
    # print(f"COWA took {time.time() - tic:.5f} seconds")
    # tic = time.time()
    # min_ise_gm = algo.fit_min_ise(gm, L=5)
    # print(f"MIN ISE took {time.time() - tic:.5f} seconds")
    # tic = time.time()
    # brute_gm = algo.fit_brute_force(gm, L=5)
    # print(f"Brute force took {time.time() - tic:.5f} seconds")

    # plot prob
    t = torch.linspace(-1, 4, steps=1000)[..., None]
    p = gm.prob(t)
    runnalls_p = runnalls_gm.prob(t).cpu()
    # west_p = west_gm.prob(t).cpu()
    runnalls2_p = runnalls2_gm.prob(t).cpu()
    # cowa_p = cowa_gm.prob(t).cpu()
    # min_ise_p = min_ise_gm.prob(t).cpu()
    # brute_p = brute_gm.prob(t).cpu()

    fh = plt.figure()
    plt.plot(t, p, "--", c="k")
    plt.plot(t, runnalls_p)
    # plt.plot(t, west_p, "--")
    plt.plot(t, runnalls2_p, "-.")
    # plt.plot(t, cowa_p, ":")
    # plt.plot(t, min_ise_p, ":")
    # plt.plot(t, brute_p, "--")
    # plt.legend(["full", "Runnalls", "West", "GMRC", "COWA", "MIN ISE", "Brute force"])
    plt.legend(["full", "Runnalls", "Runnalls2"])
    plt.xlabel("x")
    plt.ylabel("p")
    plt.show()

    fh.savefig("./images/demo.png")

    # calc error
    ise_runnalls = mixtures.calc_ise(gm, runnalls_gm)
    # ise_west = mixtures.calc_ise(gm, west_gm)
    ise_runnalls2 = mixtures.calc_ise(gm, runnalls2_gm)
    # ise_cowa = mixtures.calc_ise(gm, cowa_gm)
    # ise_min_ise = mixtures.calc_ise(gm, min_ise_gm)
    # ise_brute = mixtures.calc_ise(gm, brute_gm)

    print(f"ISE Runnalls: {ise_runnalls:.5f}")
    # print(f"ISE West: {ise_west:.5f}")
    print(f"ISE Runnalls2: {ise_runnalls2:.5f}")
    # print(f"ISE COWA: {ise_cowa:.5f}")
    # print(f"ISE MIN-ISE: {ise_min_ise:.5f}")
    # print(f"ISE Brute: {ise_brute:.5f}")


if __name__ == "__main__":
    main()
