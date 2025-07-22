import math
import scipy.stats as st


def calc_wilson_score_cc(k, phat, delta):
    z = st.norm.ppf(1 - delta / 2)
    pwsc_lower = (2 * k * phat + math.pow(z, 2) -
                  (z * (math.sqrt(math.pow(z, 2) - 1 / k + 4 * k * phat * (1 - phat) + (4 * phat - 2)) + 1))) / (
        2 * (k + math.pow(z, 2)))
    wcc_plus = max(0, pwsc_lower)
    pwsc_upper = (2 * k * phat + math.pow(z, 2) +
                  (z * (math.sqrt(math.pow(z, 2) - 1 / k + 4 * k * phat * (1 - phat) - (4 * phat - 2)) + 1))) / (
        2 * (k + math.pow(z, 2)))
    wcc_minus = min(1, pwsc_upper)
    return wcc_plus, wcc_minus


def calc_clopper_pearson(ks, k, delta):
    cp_min = st.beta.ppf(delta / 2, ks, k - ks + 1)
    cp_max = st.beta.ppf(1 - delta / 2, ks + 1, k - ks)
    return cp_min, cp_max



#print(calc_wilson_score_cc(100, 0.01, 0.05))
#print("\n")
if __name__ == '__main__':
    x, y = calc_clopper_pearson(50, 1000, 0.05)
    print((y - x) / 2)
