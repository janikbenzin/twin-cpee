import scores as sc
import numpy as np
from scipy.optimize import minimize


epsilons = [0.03, 0.025, 0.02, 0.015, 0.01, 0.005]


DELTA = 0.05

def precompute_smallest_k(eps, phat_worst):
    def clopper_pearson_epsilon(k, phat, delta, epsilon):
        # x0 is k
        ks = k * phat
        cp_min, cp_max = sc.calc_clopper_pearson(ks, k, delta)
        diff = cp_max - cp_min
        return diff / 2 - epsilon

    x0 = np.array([20])
    #res = minimize(clopper_pearson_epsilon, x0, method="nelder-mead",
    #               args=(phat_worst, DELTA, eps), options={"xatol": 1e-8, "disp": True})
    k_init = 400
    res = clopper_pearson_epsilon(k_init, phat_worst, DELTA, eps)
    knext = k_init
    while(res > 0):
        knext += 1
        res = clopper_pearson_epsilon(knext, phat_worst, DELTA, eps)
    return knext, res, sc.calc_clopper_pearson(knext * phat_worst, knext, DELTA)


if __name__ == "__main__":
    for eps in epsilons:
        print(precompute_smallest_k(eps, 0.1))
        print("\n\n")