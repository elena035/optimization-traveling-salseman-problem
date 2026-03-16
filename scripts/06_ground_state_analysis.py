import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial 
from matplotlib.ticker import PercentFormatter
from scipy.optimize import curve_fit

# -----------------------------------------
# || ARRAY INITIALIZATION ||
# -----------------------------------------

data = np.empty(8, dtype=object)

best_costs = np.zeros_like(data)
best_tours = np.zeros_like(data)
cost_history = np.zeros_like(data)
temperature_history = np.zeros_like(data)

exec_times_sa = np.zeros_like(data)
gs_probability = np.zeros_like(data)
exec_times_bf = np.zeros_like(data)
ground_state_tours = np.zeros_like(data)
ground_state_costs = np.zeros_like(data)

gs_prob_mean = np.zeros_like(data)
gs_prob_std = np.zeros_like(data)

exec_times_sa_mean = np.zeros_like(data)
exec_times_sa_std = np.zeros_like(data)

cost_history_mean = np.zeros_like(data)
cost_history_std = np.zeros_like(data)

best_cost_mean = np.zeros_like(data)
best_cost_std = np.zeros_like(data)

exec_times_bf_mean = np.zeros_like(data)
exec_times_bf_std = np.zeros_like(data)

# -----------------------------------------
# || DATA LOADING AND PROCESSING ||
# -----------------------------------------

for i, N in enumerate(range(4, 12, 1)):
    
    # Load data dynamically from the data folder
    data[i] = np.load(f"../data/gs_comparison_tsp_N{N}.npz", allow_pickle=True)
                 
    best_costs[i] = np.stack(data[i]['best_costs'])
    best_tours[i] = data[i]['best_tours'] 
    
    cost_history[i] = np.stack(data[i]['cost_history'])
    
    temperature_history[i] = np.stack(data[i]['temperature_history'])
    temperature_history[i] = temperature_history[i][0] # Assuming temperature schedule is identical for all runs
    
    exec_times_sa[i] = np.stack(data[i]['execution_times_sa'])
    gs_probability[i] = np.stack(data[i]['gs_probability'])
    
    exec_times_bf[i] = np.stack(data[i]['execution_times_bf'])
    
    ground_state_tours[i] = data[i]['tour_ground_state']
    ground_state_costs[i] = data[i]['cost_ground_state']

    # Computations (Means and Standard Deviations)
    exec_times_sa_mean[i] = np.mean(exec_times_sa[i])
    exec_times_sa_std[i] = np.std(exec_times_sa[i], ddof=1) / np.sqrt(len(exec_times_sa[i]))
    
    exec_times_bf_mean[i] = np.mean(exec_times_bf[i])
    exec_times_bf_std[i] = np.std(exec_times_bf[i], ddof=1) / np.sqrt(len(exec_times_bf[i]))
    
    best_cost_mean[i] = np.mean(best_costs[i])
    best_cost_std[i] = np.std(best_costs[i], ddof=1) / np.sqrt(len(best_costs[i]))
    
    cost_history_mean[i] = np.mean(cost_history[i], axis=0)
    cost_history_std[i] = np.std(cost_history[i], ddof=1, axis=0) / np.sqrt(len(cost_history_mean[i]))
    
    gs_prob_mean[i] = np.mean(gs_probability[i])
    gs_prob_std[i] = np.std(gs_probability[i], ddof=1) / np.sqrt(len(gs_probability[i]))

# -----------------------------------------
# || PLOT 1: PROBABILITY VS N ||
# -----------------------------------------

N_values = np.arange(4, 12, 1)

plt.figure(figsize=(12, 8))
plt.plot(N_values, gs_prob_mean, color='#6e3855', linewidth=1.5, linestyle='-')
plt.errorbar(N_values, gs_prob_mean, xerr=None, yerr=gs_prob_std, fmt='.', markersize=15, 
             color='#6e3855', ecolor='black', elinewidth=0.7, capsize=1, capthick=0.7)

plt.xlabel('\n Number of Nodes (N)', fontsize=12)
plt.ylabel('Probability (p) of finding the global minimum\n', fontsize=12)
plt.ylim(0.85, 1.01)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.grid(True, which='both', linestyle='-', color='gray', alpha=0.2)

plt.tight_layout()
plt.savefig('../assets/tsp_prob_vs_N_synthetic.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

# -----------------------------------------
# || PLOT 2: EXECUTION TIMES VS N ||
# -----------------------------------------

N_values_float = np.array(N_values, dtype=float)
exec_times_sa_mean = np.array(exec_times_sa_mean, dtype=float)
exec_times_sa_std = np.array(exec_times_sa_std, dtype=float)

# Define the power-law model for Simulated Annealing execution time
def power_law(t, a, m):
    return a * t**m

# Fit with initial estimation
popt, pcov = curve_fit(power_law, N_values_float, exec_times_sa_mean, sigma=exec_times_sa_std, absolute_sigma=True, p0=(1, 1))
a, m = popt
a_err, m_err = np.sqrt(np.diag(pcov))

eq_str = r"$ t = (%.3f \pm %.3f) \cdot N^{%.3f \pm %.3f}$" % (a, a_err, m, m_err)

plt.figure(figsize=(12, 8))

# Simulated Annealing Line & Errors
plt.plot(N_values, exec_times_sa_mean, color='#6e3855', linewidth=1.5, linestyle='-', label=f'Simulated Annealing: {eq_str}')
plt.errorbar(N_values, exec_times_sa_mean, xerr=None, yerr=exec_times_sa_std, fmt='.', markersize=15, 
             color='#6e3855', ecolor='black', elinewidth=0.7, capsize=1, capthick=0.7)

# Exact Algorithm (Brute Force) Line & Errors
plt.plot(N_values, exec_times_bf_mean, color='#ad3e79', linewidth=1.5, linestyle='-', label=r'Exact Algorithm: $t = 3.5 \cdot 10^{-6} (N - 1)!$')
plt.errorbar(N_values, exec_times_bf_mean, xerr=None, yerr=exec_times_bf_std, fmt='.', markersize=15, 
             color='#ad3e79', ecolor='black', elinewidth=0.7, capsize=1, capthick=0.7)

plt.xlabel('\n Number of Nodes (N)', fontsize=12)
plt.ylabel('Execution Time t [s] \n', fontsize=12)
plt.grid(True, which='both', linestyle='-', color='gray', alpha=0.2)
plt.legend(fontsize=10, labelspacing=1)

plt.tight_layout()
plt.savefig('../assets/tsp_time_vs_N_synthetic_fit.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()