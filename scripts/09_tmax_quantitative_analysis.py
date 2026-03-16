import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import PercentFormatter

# -----------------------------------------
# || CONFIGURATION & ARRAY INITIALIZATION ||
# -----------------------------------------

T_max_values = [75, 125, 250, 500, 1000, 1500, 2000, 3000, 4000, 8000, 16000, 32000, 64000, 128000]
n_points = len(T_max_values)

data = np.empty(n_points, dtype=object)

best_costs = np.zeros(n_points, dtype=object)
best_costs_mean = np.zeros(n_points)
best_costs_std = np.zeros(n_points)
optimal_cost = np.zeros(n_points)

# -----------------------------------------
# || DATA LOADING AND PROCESSING ||
# -----------------------------------------

for i, T_max in enumerate(T_max_values):
    # Load 10-run data
    data_10run = np.load(f'../data/T_ww/dati_tsp_ww_T{T_max}_10run.npz', allow_pickle=True)
    
    # Load 5-run data
    data_5run = np.load(f'../data/T_ww/dati_tsp_ww_T{T_max}_5run.npz', allow_pickle=True)
    
    # Combine the two datasets to get 15 runs total
    combined_data = {}
    for key in data_10run.files:
        combined_data[key] = np.concatenate([data_10run[key], data_5run[key]], axis=0)
        
    data[i] = combined_data
             
    # Extract best costs array
    best_costs[i] = np.stack(data[i]['best_costs'])
    
    # Find the minimum cost reached in each individual run
    best_costs[i] = np.min(best_costs[i], axis=1)
    
    # Compute absolute minimum, mean, and standard error of the mean
    optimal_cost[i] = min(best_costs[i])
    best_costs_mean[i] = np.mean(best_costs[i])
    best_costs_std[i] = np.std(best_costs[i], ddof=1) / np.sqrt(len(best_costs[i]))

# Save aggregated results ONCE after the loop
np.savez('../data/aggregated_best_costs_Tmax_ww_15runs.npz',
         T_max_values=T_max_values,
         best_cost_mean=best_costs_mean,
         best_cost_std=best_costs_std,
         optimal_cost=optimal_cost)


# -----------------------------------------
# || PLOTTING ||
# -----------------------------------------

fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot mean best costs with error bars
ax1.plot(T_max_values, best_costs_mean, '#6e3855', linewidth=1.5, label='Mean of minimum costs recorded per run')
ax1.errorbar(T_max_values, best_costs_mean, xerr=None, yerr=best_costs_std, fmt='.', markersize=15, 
             color='#6e3855', ecolor='#6e3855', elinewidth=0.7, capsize=1, capthick=0.7)

# Formatting axes
ax1.set_xlabel(r"Maximum Initial Temperature $T_{\mathrm{max}}$", fontsize=12)
ax1.set_ylabel("Minimum Cost [km]\n", fontsize=12)
ax1.set_xscale('log') # Logarithmic scale for the wide range of temperatures

# Set Y-axis to scientific notation
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))
ax1.set_ylim(5.6 * 10**5, 7.2 * 10**5)
ax1.yaxis.set_major_formatter(formatter)

# Grid and Legend
ax1.grid(True, which='both', linestyle='-', color='gray', alpha=0.2)
plt.legend(fontsize=12, loc='lower right')

plt.tight_layout()

# Save the figure
plt.savefig('../assets/tsp_sa_ww_Tmax_range_15runs_min_cost.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()