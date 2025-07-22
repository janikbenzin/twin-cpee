import json
import collections as cl
from scores import calc_clopper_pearson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reliability_assurance_quantification import precompute_smallest_k, DELTA

EXPERIMENTS = ["heating_batch_81", "heating_batch_90", "heating_batch_99", "waterlevel_batch_11",
               "waterlevel_batch_20", "waterlevel_batch_29", "irrigation_batch_1"]
exp = EXPERIMENTS[0]
TIME = "time"
TEMP = "temperature"
LEVEL = "level"
VAR = "variable"
SUC = "empirical_success"
EPS = "epsilon"
PHAT_WORST = 0.1

def visualize_scenarios(df, scenario, ci=False, format="svg", filename=None, xticklabels=None,
                        legend_tite="Dependence on", suptitle=None, boundary_label="Temperature (°C) "):
    g = sns.FacetGrid(df[df['scenario'] == scenario],
                      col='config',
                      hue='config',
                      height=4, aspect=1.2, sharey=True, sharex=False)

    def plot_with_ci(data, **kwargs):
        ax = plt.gca()
        for config in data['config'].unique():
            sub = data[data['config'] == config]
            ax.plot(sub['boundary_value'], sub['empirical_prob'], **kwargs)
            if ci:
                ax.fill_between(sub['boundary_value'], sub['ci_lower'], sub['ci_upper'], alpha=0.5)
            if xticklabels is not None and "simultaneously" in config:
                ax.set_xticklabels(xticklabels)

    g.map_dataframe(plot_with_ci)
    g.set_axis_labels(boundary_label + "boundary value", "Empirical probability")

    g.set_titles(col_template="")
    g.add_legend(title=legend_tite)
    g._legend.set_bbox_to_anchor((0.94, 0.7))  # (x, y) relative to the whole figure
    #g._legend.set_frame_on(True)  # Optional: add border
    #g._legend.set_borderaxespad(0.0)
    suptitle = f"Empirical probability vs. boundary value for {scenario}" if suptitle is None else suptitle
    g.fig.suptitle(suptitle, y=0.95)
    plt.tight_layout()
    plt.savefig(f"{scenario if filename is None else filename}_{'ci' if ci else 'wo_ci'}.{format}")
    plt.subplots_adjust(right=0.75)
    plt.close()


def eval_emp_heating_boundaries(results, exp, bound, eval, bound2=None):
    if bound2 is None:
        return [1 if any([eval(temp, bound) for temp in it["variable"]]) else 0 for sid, it in results[exp].items()]
    else:
        return [1 if any([eval(temp, bound, bound2) for temp in it["variable"]]) else 0 for sid, it in
                results[exp].items()]


def get_epsilon_from_clopper_pearson(ks, k, delta):
    lower, upper = calc_clopper_pearson(ks, k, 0.05)
    return (upper - lower) / 2


def eval_single_success_epsilon(res, exp, bound, successes, k):
    x = eval_emp_heating_boundaries(res, exp, bound, lambda t, b: t < b)
    c = cl.Counter(x)
    emp_success_rate = c[1] / k
    successes[exp][SUC].append(emp_success_rate)
    successes[exp][EPS].append(get_epsilon_from_clopper_pearson(c[1], k, DELTA))


def get_multi_success_dict():
    return {SUC:
                {upper_partial: [],
                 lower_partial: [],
                 upper_lower: []
                 },
            EPS:
                {upper_partial: [],
                 lower_partial: [],
                 upper_lower: []
                 }
            }



def eval_multi_success_epsilon(res, exp, bound1, bound2, successes, mode, eval, k):
    x = eval_emp_heating_boundaries(res, exp, bound1, eval, bound2)
    c = cl.Counter(x)
    emp_success_rate = c[1] / k
    successes[exp][SUC][mode].append(emp_success_rate)
    successes[exp][EPS][mode].append(get_epsilon_from_clopper_pearson(c[1], k, DELTA))



with open("results/results.json", "r") as f:
    res = json.load(f)


step_width = 0.1
heating_upper_bound = 80
heating_lower_bound = 70
heating_steps = (heating_upper_bound - heating_lower_bound) / step_width
heating_boundaries = [heating_lower_bound + i * step_width for i in range(int(heating_steps) + 1)]




successes = {exp: {SUC: [],
                   EPS: []} for exp in EXPERIMENTS[:3]}
k = 1000
for exp in EXPERIMENTS[:3]:
    for bound in heating_boundaries:
        eval_single_success_epsilon(res, exp, bound, successes, k)


lower_one_sided = "low"
configs = [lower_one_sided]

boundary_values = {lower_one_sided: heating_boundaries}

data = []

for exp in EXPERIMENTS[:3]:
    config = configs[0]
    for i, b_val in enumerate(boundary_values[config]):
        if successes[exp][SUC][i] > 0 and successes[exp][EPS][i] > 0:
            data.append({
                'scenario': config,
                'config': exp,
                'boundary_value': b_val,
                'empirical_prob': successes[exp][SUC][i],
                'ci_lower': successes[exp][SUC][i] - successes[exp][EPS][i] if successes[exp][SUC][i] -
                                                                               successes[exp][EPS][i] > 0 else 0,
                'ci_upper': successes[exp][SUC][i] + successes[exp][EPS][i]
            })
        elif successes[exp][SUC][i] == 0 or successes[exp][EPS][i] < 0:
            data.append({
                'scenario': config,
                'config': exp,
                'boundary_value': b_val,
                'empirical_prob': successes[exp][SUC][i],
                'ci_lower': np.nan,
                'ci_upper': np.nan
            })

df_h = pd.DataFrame(data)
# Above 10% observed empirical probability our assumption of phat worst lower than 10% resulting in less than 1k runs is violated, i.e., confidence intervals cannot be trusted anymore
df_h_f = df_h[df_h["empirical_prob"] < PHAT_WORST]

plt.figure(figsize=(10, 30))
sns.set(style='whitegrid', rc={"figure.dpi": 192})

visualize_scenarios(df_h, configs[0], ci=False, format="png", filename="heating", legend_tite="Scenario",
                    suptitle="Empirical probability vs. boundary value")
plt.figure(figsize=(10, 30))
visualize_scenarios(df_h_f, configs[0], ci=True, format="png", filename="heating", legend_tite="Scenario",
                    suptitle="Empirical probability vs. boundary value")

df_h_f = df_h[df_h["empirical_prob"] < PHAT_WORST]

### Waterlevel

wl_upper_bound_min = 30
wl_upper_bound_max = 39
wl_upper_bound_max_t = 39
wl_lower_bound_max = 10
wl_lower_bound_min = 1
wl_steps_lower = (wl_lower_bound_max - wl_lower_bound_min) / step_width
wl_steps_upper = (wl_upper_bound_max - wl_upper_bound_min) / step_width
wl_steps_upper_t = (wl_upper_bound_max_t - wl_upper_bound_min) / step_width
wl_upper_boundaries = [wl_upper_bound_min + i * step_width for i in range(int(wl_steps_upper) + 1)]
wl_lower_boundaries = [wl_lower_bound_min + i * step_width for i in range(int(wl_steps_lower) + 1)]
wl_upper_boundaries_t = [wl_upper_bound_min + i * step_width for i in range(int(wl_steps_upper_t) + 1)]
wl_upper_boundaries_t.reverse()

upper_partial = "u"
lower_partial = "l"
upper_lower = "l and u simultaneously"

for exp in EXPERIMENTS[3:-1]:
    successes[exp] = get_multi_success_dict()


for exp in EXPERIMENTS[3:-1]:
    for bound in wl_upper_boundaries:
        eval_multi_success_epsilon(res, exp, bound, None, successes, upper_partial, lambda t, b: t > b, k)
    for bound in wl_lower_boundaries:
        eval_multi_success_epsilon(res, exp, bound, None, successes, lower_partial, lambda t, b: t < b, k)
    for bound_l, bound_u in zip(wl_lower_boundaries, wl_upper_boundaries_t):
        eval_multi_success_epsilon(res, exp, bound_l, bound_u, successes, upper_lower,
                                   lambda t, b1, b2: t < b1 or t > b2, k)

configs = [lower_partial, upper_partial, upper_lower]
boundary_values = {lower_partial: wl_lower_boundaries, upper_partial: wl_upper_boundaries,
                   upper_lower: wl_upper_boundaries_t}
#ticks = [f"{round(low,1)}, {round(up,1)}" for low, up in zip(wl_lower_boundaries, wl_upper_boundaries_t)]
ticks = ["(2.0, 38.0)", "(4.0, 36.0)", "(6.0, 34.0)", "(8.0, 32.0)", "(10.0, 30.0)"]
ticks.reverse()
ticks = [""] + ticks
ticks_11 = ["(1.0, 39.0)", "(1.5, 38.5)", "(2.0, 38.0)", "(2.5, 37.5)", "(3.0, 37.0)"]
ticks_11.reverse()
ticks_11 = [""] + ticks_11
ticks_20 = ["(1.0, 39.0)", "(2.0, 38.0)", "(3.0, 37.0)", "(4.0, 36.0)", "(5.0, 35.0)"]
ticks_20.reverse()
ticks_20 = [""] + ticks_20
ticks_29 = ["(1.0, 39.0)", "(2.0, 38.0)", "(3.0, 37.0)", "(4.0, 36.0)", "(5.0, 35.0)", "(4.0, 34.0)"]
ticks_29.reverse()
ticks_29 = [""] + ticks_29
data = []

for exp in EXPERIMENTS[3:-1]:
    for config in configs:
        for i, b_val in enumerate(boundary_values[config]):
            if successes[exp][SUC][config][i] > 0 and successes[exp][EPS][config][i] > 0:
                data.append({
                    'scenario': exp,
                    'config': config,
                    'boundary_value': b_val,
                    'empirical_prob': successes[exp][SUC][config][i],
                    'ci_lower': successes[exp][SUC][config][i] - successes[exp][EPS][config][i] if
                    successes[exp][SUC][config][i] - successes[exp][EPS][config][i] > 0 else 0,
                    'ci_upper': successes[exp][SUC][config][i] + successes[exp][EPS][config][i]
                })
            elif successes[exp][SUC][config][i] == 0 or successes[exp][EPS][config][i] < 0:
                data.append({
                    'scenario': exp,
                    'config': config,
                    'boundary_value': b_val,
                    'empirical_prob': successes[exp][SUC][config][i],
                    'ci_lower': np.nan,
                    'ci_upper': np.nan
                })

df_wl = pd.DataFrame(data)
df_wl_f = df_wl[df_wl["empirical_prob"] < PHAT_WORST]
plt.figure(figsize=(10, 30))
sns.set(style='whitegrid', rc={"figure.dpi": 192})



for scenario in df_wl['scenario'].unique():
    visualize_scenarios(df_wl, scenario, ci=False, format="png", xticklabels=ticks, boundary_label="Water level (l) ")
    plt.figure(figsize=(10, 30))
    if scenario.endswith("11"):
        visualize_scenarios(df_wl_f, scenario, ci=True, format="png", xticklabels=ticks_11, boundary_label="Water level (l) ")
    elif scenario.endswith("20"):
        visualize_scenarios(df_wl_f, scenario, ci=True, format="png", xticklabels=ticks_20, boundary_label="Water level (l) ")
    else:
        visualize_scenarios(df_wl_f, scenario, ci=True, format="png", xticklabels=ticks_29, boundary_label="Water level (l) ")
    plt.figure(figsize=(10, 30))

    #visualize_scenarios(df_waterlevel, scenario, format="png")
    #plt.figure(figsize=(10, 30))



### IRRIGATION
step_width = 0.005

## Only too try soil can be the responsibility of the irrigation systems, as rain can increase the moisture

#irr_upper_bound_min = 0.4
#irr_upper_bound_max = 0.6
irr_lower_bound_max = 0.21
irr_lower_bound_min = 0.01
irr_steps_lower = (irr_lower_bound_max - irr_lower_bound_min) / step_width
#irr_steps_upper = (irr_upper_bound_max - irr_upper_bound_min) / step_width
#irr_upper_boundaries = [irr_upper_bound_min + i * step_width for i in range(int(irr_steps_upper) + 1)]
irr_lower_boundaries = [round(irr_lower_bound_min + i * step_width, 3) for i in range(int(irr_steps_lower) + 1)]
#irr_upper_boundaries.reverse()


successes[EXPERIMENTS[-1]] = {SUC: [], EPS: []}

for exp in EXPERIMENTS[-1:]:
    #for bound in irr_upper_boundaries:
    #    eval_multi_success_epsilon(res, exp, bound, None, successes, upper_partial, lambda t, b: t > b, k)
    for bound in irr_lower_boundaries:
        eval_single_success_epsilon(res, exp, bound, successes, k)
    #for bound_l, bound_u in zip(irr_lower_boundaries, irr_upper_boundaries):
    #    eval_multi_success_epsilon(res, exp, bound_l, bound_u, successes, upper_lower,
    #                               lambda t, b1, b2: t < b1 or t > b2, k)

#configs = [lower_partial, upper_partial, upper_lower]
configs = [lower_one_sided]

boundary_values = {lower_one_sided: irr_lower_boundaries}

for exp in EXPERIMENTS[-1:]:
    config = configs[0]
    for i, b_val in enumerate(boundary_values[config]):
        if successes[exp][SUC][i] > 0 and successes[exp][EPS][i] > 0:
            data.append({
                'scenario': config,
                'config': exp,
                'boundary_value': b_val,
                'empirical_prob': successes[exp][SUC][i],
                'ci_lower': successes[exp][SUC][i] - successes[exp][EPS][i] if successes[exp][SUC][i] -
                                                                               successes[exp][EPS][i] > 0 else 0,
                'ci_upper': successes[exp][SUC][i] + successes[exp][EPS][i]
            })
        elif successes[exp][SUC][i] == 0 or successes[exp][EPS][i] < 0:
            data.append({
                'scenario': config,
                'config': exp,
                'boundary_value': b_val,
                'empirical_prob': successes[exp][SUC][i],
                'ci_lower': np.nan,
                'ci_upper': np.nan
            })

df_irr = pd.DataFrame(data)
# Above 10% observed empirical probability our assumption of phat worst lower than 10% resulting in less than 1k runs is violated, i.e., confidence intervals cannot be trusted anymore
df_irr_f = df_irr[df_irr["empirical_prob"] < PHAT_WORST]

plt.figure(figsize=(10, 30))
sns.set(style='whitegrid', rc={"figure.dpi": 192})

visualize_scenarios(df_irr, configs[0], ci=False, format="png", filename="irrigation", legend_tite="Scenario",
                    suptitle="Empirical probability vs. boundary value", boundary_label="Moisture (% per m3) ")
plt.figure(figsize=(10, 30))
visualize_scenarios(df_irr_f, configs[0], ci=True, format="png", filename="irrigation", legend_tite="Scenario",
                    suptitle="Empirical probability vs. boundary value",  boundary_label="Moisture (% per m3) ")

df_irr_f = df_irr[df_irr["empirical_prob"] < PHAT_WORST]


### Sequential

epsilons = [0.025, 0.02, 0.015, 0.01, 0.005]
allowed_epsilons = []
i = 0
k = precompute_smallest_k(epsilons[i], PHAT_WORST)[0]
allowed_epsilons.append((epsilons[i], k))
while k < 1000:
    i += 1
    k = precompute_smallest_k(epsilons[i], PHAT_WORST)[0]
    if k < 1000:
        allowed_epsilons.append((epsilons[i], k))

# Now we set the boundary value such that reliable operation is guaranteed

heating_boundary = 76  # with prob 1.3% (81), 3.0% (90), 2.3% (99) -> conservative 3.0% [0.018890, 0.041110]
wl_boundary = (2, 38)  # with prob 4.2% (11), 0.3% (20), 0.1% (29) -> conservative 4.2% [0.029042, 0.054958]
irr_boundary = 0.18  # with prob 0.4%

fixed_heating_probability = [successes[exp][SUC][int(next(i for i, x in enumerate(heating_boundaries) if x == heating_boundary))] for exp in EXPERIMENTS[:3]]
print(f"Fixed-k - The water kettle system shows the following empirically estimated probabilities in the order from 81 to 99 initial parameter: {fixed_heating_probability}\n\n\n")
fixed_heating_real_eps = [successes[exp][EPS][int(next(i for i, x in enumerate(heating_boundaries) if x == heating_boundary))] for exp in EXPERIMENTS[:3]]
print(
    f"Fixed-k - The water kettle system shows the following confidence intervals: {fixed_heating_real_eps}\n\n\n")

fixed_wl_probability = [successes[exp][SUC][upper_lower][int(next(i for i, x in enumerate(wl_upper_boundaries_t) if x == wl_boundary[1]))] for exp in EXPERIMENTS[3:-1]]
print(f"Fixed-k - The water tank system shows the following empirically estimated probabilities in the order from 11 to 29 initial parameter: {fixed_wl_probability}\n\n\n")
fixed_wl_real_eps = [successes[exp][EPS][upper_lower][int(next(i for i, x in enumerate(wl_upper_boundaries_t) if x == wl_boundary[1]))] for exp in EXPERIMENTS[3:-1]]
print(
    f"Fixed-k - The water tank system shows the following confidence intervals: {fixed_wl_real_eps}\n\n\n")

fixed_irr_probability = [successes[exp][SUC][int(next(i for i, x in enumerate(irr_lower_boundaries) if x == irr_boundary))] for exp in EXPERIMENTS[-1:]]
print(f"Fixed-k - The irrigation system shows the following empirically estimated probability: {fixed_irr_probability}\n\n\n")
fixed_irr_eps = [successes[exp][EPS][int(next(i for i, x in enumerate(irr_lower_boundaries) if x == irr_boundary))] for exp in EXPERIMENTS[-1:]]
print(
    f"Fixed-k - The irrigation system shows the following confidence intervals: {fixed_irr_eps}\n\n\n")

## OVERALL on/off or heat/cool nondeterministically decided
first_k_successes = {k:
                         {exp: {SUC: [], EPS: []} for exp in EXPERIMENTS[:3] + EXPERIMENTS[-1:]} for eps, k in allowed_epsilons}
first_k_results = {k: {} for eps, k in allowed_epsilons}
assumption_holds = {k: [True, True, True] for eps, k in allowed_epsilons}  # heating, waterlevel, irrigation

for eps, k in allowed_epsilons:
    first_k_results[k] = {exp:
                              {sid: res[exp][sid] for i, sid in enumerate(res[exp].keys()) if i < k}
                          for exp in EXPERIMENTS}

    for exp in EXPERIMENTS[:3]:
        eval_single_success_epsilon(res, exp, heating_boundary, first_k_successes[k], k)
        # Check whether the phat worst assumption holds (necessary since we have not precomputed k with the always worst phat
        if first_k_successes[k][exp][SUC][0] > PHAT_WORST:
            assumption_holds[k][0] = False

    for exp in EXPERIMENTS[3:-1]:
        first_k_successes[k][exp] = get_multi_success_dict()
        eval_multi_success_epsilon(res, exp, wl_boundary[0], wl_boundary[1], first_k_successes[k], upper_lower,
                                   lambda t, b1, b2: t < b1 or t > b2, k)
        if first_k_successes[k][exp][SUC][upper_lower][0] > PHAT_WORST:
            assumption_holds[k][1] = False

    for exp in EXPERIMENTS[-1:]:
        eval_single_success_epsilon(res, exp, irr_boundary, first_k_successes[k], k)
        # Check whether the phat worst assumption holds (necessary since we have not precomputed k with the always worst phat
        if first_k_successes[k][exp][SUC][0] > PHAT_WORST:
            assumption_holds[k][2] = False

# Find smallest k such that assumption holds
smallest_k_heating = min([k for k in assumption_holds.keys() if assumption_holds[k][0]])
smallest_k_wl = min([k for k in assumption_holds.keys() if assumption_holds[k][1]])
smallest_k_irr = min([k for k in assumption_holds.keys() if assumption_holds[k][2]])

eps, k = allowed_epsilons[0]
if smallest_k_heating == k and smallest_k_wl == k and smallest_k_irr == k:
    # No assumption is violated

    # As assumption_holds_interval is always True, we can compute for the respective epsilon
    ep, k = allowed_epsilons[0]
    prec_heating_probability_025 = [first_k_successes[k][exp][SUC][0] for exp in EXPERIMENTS[:3]]
    print(f"Sequential 2.5% - The water kettle system shows the following empirically estimated probabilities in the order from 81 to 99 initial parameter: {prec_heating_probability_025}\n\n\n")
    prec_heating_real_eps_025 = [first_k_successes[k][exp][EPS][0] for exp in EXPERIMENTS[:3]]
    print(
        f"Sequential 2.5% - The water kettle system shows the following confidence intervals: {prec_heating_real_eps_025}\n\n\n")

    prec_wl_probability_025 = [first_k_successes[k][exp][SUC][upper_lower][0] for exp in EXPERIMENTS[3:-1]]
    print(
        f"Sequential 2.5% - The water tank system shows the following empirically estimated probabilities in the order from 11 to 29 initial parameter: {prec_wl_probability_025}\n\n\n")
    prec_wl_real_eps_025 = [first_k_successes[k][exp][EPS][upper_lower][0] for exp in EXPERIMENTS[3:-1]]
    print(
        f"Sequential 2.5% - The water tank system shows the following confidence intervals: {prec_wl_real_eps_025}\n\n\n")

    prec_irr_probability_025 = [first_k_successes[k][exp][SUC][0] for exp in EXPERIMENTS[-1:]]
    print(
        f"Sequential 2.5% - The irrigiation system shows the following empirically estimated probability: {prec_irr_probability_025}\n\n\n")
    prec_irr_real_eps_025 = [first_k_successes[k][exp][EPS][0] for exp in EXPERIMENTS[-1:]]
    print(
        f"Sequential 2.5% - The irrigiation system shows the following confidence intervals: {prec_irr_real_eps_025}\n\n\n")

    ep, k = allowed_epsilons[1]
    prec_heating_probability_02 = [first_k_successes[k][exp][SUC][0] for exp in EXPERIMENTS[:3]]
    print(
        f"Sequential 2.0% - The water kettle system shows the following empirically estimated probabilities in the order from 81 to 99 initial parameter: {prec_heating_probability_02}\n\n\n")
    prec_heating_real_eps_02 = [first_k_successes[k][exp][EPS][0] for exp in EXPERIMENTS[:3]]
    print(
        f"Sequential 2.0% - The water kettle system shows the following confidence intervals: {prec_heating_real_eps_02}\n\n\n")

    prec_wl_probability_02 = [first_k_successes[k][exp][SUC][upper_lower][0] for exp in EXPERIMENTS[3:-1]]
    print(
        f"Sequential 2.0% - The water kettle system shows the following empirically estimated probabilities in the order from 11 to 29 initial parameter: {prec_wl_probability_02}\n\n\n")
    prec_wl_real_eps_02 = [first_k_successes[k][exp][EPS][upper_lower][0] for exp in EXPERIMENTS[3:-1]]
    print(
        f"Sequential 2.0% - The water kettle system shows the following confidence intervals: {prec_wl_real_eps_02}\n\n\n")

    prec_irr_probability_02 = [first_k_successes[k][exp][SUC][0] for exp in EXPERIMENTS[-1:]]
    print(
        f"Sequential 2.0% - The water kettle system shows the following empirically estimated probability: {prec_irr_probability_02}\n\n\n")
    prec_irr_real_eps_02 = [first_k_successes[k][exp][EPS][0] for exp in EXPERIMENTS[-1:]]
    print(
        f"Sequential 2.0% - The water kettle system shows the following confidence intervals: {prec_irr_real_eps_02}\n\n\n")



