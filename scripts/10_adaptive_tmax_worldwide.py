import numpy as np
import random
from tqdm import tqdm

# -----------------------------------------
# || CONFIGURATION & VARIABLES ||
# -----------------------------------------

runs = 500
beta_max = 2000 # Corresponds to T_min = 0.0005
epsilon = 0.0001 # Prevents division by zero

# Load the distance matrix for the 500 worldwide cities (saved from script 08)
data = np.load('../data/distance_matrix_ww_500.npz')
distances = data['distance_matrix']

N = len(distances) # Number of cities to visit

# -----------------------------------------
# || CORE FUNCTION ||
# -----------------------------------------

def calculate_initial_beta(initial_tour, distance_matrix, target_acceptance):
    """
    Estimates the optimal initial beta (which gives T_max) by sampling 
    random swaps and calculating the average cost variation (Delta C).
    We want the probability of accepting worse solutions at T_max to be 
    equal to 'target_acceptance'.
    """
    deltas = []
    for _ in range(1000):
        i, j = random.sample(range(N), 2)
        t_new = initial_tour[:]
        t_new[i], t_new[j] = t_new[j], t_new[i]
        
        c_new = sum(distance_matrix[t_new[k], t_new[(k+1)%N]] for k in range(N))
        c_old = sum(distance_matrix[initial_tour[k], initial_tour[(k+1)%N]] for k in range(N))
        
        delta = c_new - c_old
        deltas.append(abs(delta))
    
    # Average of absolute cost variations
    mean_delta = np.mean(deltas)
    
    # From Metropolis: target_acceptance = exp(-beta * mean_delta)
    # Therefore: beta = -ln(target_acceptance) / mean_delta
    beta = - np.log(target_acceptance) / (mean_delta + epsilon)
    
    return beta

# -----------------------------------------
# || MAIN LOOP OVER ACCEPTANCE RATES ||
# -----------------------------------------

# Test target acceptance rates from 10% to 25% (step 5%)
for i in tqdm(range(10, 30, 5), desc="Acceptance Rates Progress"):
    acceptance_pct = i
    target_acceptance = i / 100.0  # e.g., 0.10, 0.15, 0.20, 0.25
    
    beta_min_total = np.empty((runs,), dtype=object)
    
    for run_idx in range(runs):
        nodes = list(range(1, N))
        random.shuffle(nodes)
        start_tour = [0] + nodes # Fix the first node
        
        # Calculate beta using the localized target_acceptance
        initial_beta = calculate_initial_beta(start_tour, distances, target_acceptance)
        
        beta_min_total[run_idx] = initial_beta
    
    # Save the estimated beta_mins for this specific acceptance rate
    np.savez(f'../data/T_ww/adaptive_Tmax_ww_acc{acceptance_pct}.npz',
             beta_max=beta_max,
             beta_min=beta_min_total)