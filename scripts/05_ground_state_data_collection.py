import numpy as np
import random
from itertools import permutations
from tqdm import tqdm
import time

# -----------------------------------------
# || GENERAL SETTINGS & VARIABLES ||
# -----------------------------------------

annealing_cycles = 4
beta_max = 1000

cooling_rate = 1.02 # Decreases temperature by 2%
heating_rate = 1.05 # Increases temperature by 5%

# -----------------------------------------
# || FUNCTION DEFINITIONS ||
# -----------------------------------------

def exact_ground_state(distance_matrix):
    """Finds the absolute minimum cost using Brute Force permutations."""
    visited_tours = []
    costs = []
    
    # We fix the first node (0) to get (N-1)! permutations and avoid duplicates
    for perm in permutations(range(1, N)):
        tour = [0] + list(perm) 
        
        cost = sum(distance_matrix[tour[j], tour[(j + 1) % len(tour)]] for j in range(len(tour)))
        visited_tours.append(tour)
        costs.append(cost)
        
    cost_ground_state = min(costs)
    idx_ground_state = np.argmin(costs)
    tour_ground_state = visited_tours[idx_ground_state]
    
    return tour_ground_state, cost_ground_state


def cooling(distance_matrix, initial_tour, initial_beta):
    current_tour = initial_tour[:]
    current_cost = sum(distance_matrix[current_tour[i], current_tour[(i+1)%len(current_tour)]] for i in range(len(current_tour)))
    
    best_tour = current_tour[:]
    best_cost = current_cost
    
    beta = initial_beta
    T = 1 / beta
    
    cost_history = [current_cost]
    tour_history = [current_tour]
    temperature_history = [T]
    
    step = 1
    while True:
        # Swap two random cities (avoiding the fixed 0-th city)
        i, j = random.sample(range(1, N), 2) 
        new_tour = current_tour[:]
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i] 
        
        new_cost = sum(distance_matrix[new_tour[i], new_tour[(i+1)%len(current_tour)]] for i in range(len(new_tour)))
        delta = new_cost - current_cost 

        if delta < 0 or random.random() < np.exp(-beta * delta):
            current_tour = new_tour
            current_cost = new_cost

            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = current_tour[:]
                
        if step % save_interval == 0: 
            cost_history.append(current_cost)
            tour_history.append(current_tour)

        if step % beta_update_interval_c == 0:
            beta = min(beta * cooling_rate, beta_max)
            T = 1 / beta
            temperature_history.append(T)

        if beta >= beta_max:
            break
        step += 1
        
    return best_cost, best_tour, cost_history, tour_history, temperature_history


def heating(distance_matrix, initial_tour, initial_beta):
    current_tour = initial_tour[:]
    current_cost = sum(distance_matrix[current_tour[i], current_tour[(i+1)%len(current_tour)]] for i in range(len(current_tour)))
    
    best_tour = current_tour[:]
    best_cost = current_cost
    
    beta = max(initial_beta / heating_rate, beta_min)
    T = 1 / beta
    
    cost_history = [current_cost]
    tour_history = [current_tour]
    temperature_history = [T]
    
    step = 1
    while True:
        i, j = random.sample(range(1, N), 2) 
        new_tour = current_tour[:]
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i] 
        
        new_cost = sum(distance_matrix[new_tour[i], new_tour[(i+1)%len(current_tour)]] for i in range(len(new_tour)))
        delta = new_cost - current_cost 

        if delta < 0 or random.random() < np.exp(-beta * delta):
            current_tour = new_tour
            current_cost = new_cost

            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = current_tour[:]
                
        if step % save_interval == 0: 
            cost_history.append(current_cost)
            tour_history.append(current_tour)

        if step % beta_update_interval_h == 0:
            beta = max(beta / heating_rate, beta_min) 
            T = 1 / beta
            temperature_history.append(T)

        if beta <= beta_min:
            break
        step += 1
        
    return best_cost, best_tour, cost_history, tour_history, temperature_history

# -----------------------------------------
# || MAIN LOOP OVER N ||
# -----------------------------------------

for n in [4, 5, 6, 7, 8, 9, 10, 11]:
    print(f'\n--- Processing N = {n} ---')
    
    N = n
    
    # Adaptive beta based on N
    if N == 4:
        beta_min = 0.008
    elif 4 < N < 8:
        beta_min = 0.006
    elif 8 <= N < 20:
        beta_min = 0.003
    else:
        beta_min = 0.002
        
    # Dynamically set runs and intervals
    if N < 10:
        runs = 300
    else: 
        runs = 100
        
    if N < 150:
        beta_update_interval_c = 100 
        beta_update_interval_h = 100 
        save_interval = 100
    else:
        beta_update_interval_c = 300 
        beta_update_interval_h = 300 
        save_interval = 300
        
    # Initialize storage arrays
    best_cost_tot = np.empty((runs,), dtype=object)
    best_tour_tot = np.empty((runs,), dtype=object)
    cost_history_tot = np.empty((runs,), dtype=object)
    tour_history_tot = np.empty((runs,), dtype=object)
    temperature_history_tot = np.empty((runs,), dtype=object)
    
    gs_probability = np.empty((runs,), dtype=object)
    
    execution_times_sa = np.empty((runs,), dtype=object)
    execution_times_bf = np.empty((runs,), dtype=object)
    
    tour_ground_state_tot = np.empty((runs,), dtype=object)
    cost_ground_state_tot = np.empty((runs,), dtype=object)
    
    for run_idx in tqdm(range(runs), desc="Runs Progress"):
        
        # --- SYNTHETIC MODULAR DISTRIBUTION SETUP ---
        num_clusters = int(np.sqrt(N))
        points_per_cluster = N // num_clusters
        cluster_std = np.sqrt(N) / 20 
    
        cluster_centers = np.random.rand(num_clusters, 2) * np.sqrt(N) 
    
        points = []
        for center in cluster_centers:
            cluster_points = np.random.randn(points_per_cluster, 2) * cluster_std + center     
            points.append(cluster_points)
    
        points = np.vstack(points) 
    
        while len(points) < N:
            extra = np.random.randn(1, 2) * cluster_std + cluster_centers[0]
            points = np.vstack([points, extra])
    
        distances = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(points[i] - points[j]) 
                distances[i, j] = dist
                distances[j, i] = dist 
        
        # --- EXACT ALGORITHM (Brute Force) ---
        start_bf = time.time()
        tour_ground_state, cost_ground_state = exact_ground_state(distances)
        end_bf = time.time()
    
        time_bf = end_bf - start_bf
        
        tour_ground_state_tot[run_idx] = tour_ground_state
        cost_ground_state_tot[run_idx] = cost_ground_state
        execution_times_bf[run_idx] = time_bf
    
        # --- SIMULATED ANNEALING ---
        best_cost = []
        best_tour = []
        cost_history = []
        tour_history = []
        temperature_history = []
        
        # Initialization
        nodes = list(range(1, N))
        random.shuffle(nodes)
        start_tour = [0] + nodes # Fix the first node
        
        best_cost_c, best_tour_c, cost_hist_c, tour_hist_c, temp_hist_c = cooling(distances, start_tour, beta_min)
        best_cost_h, best_tour_h, cost_hist_h, tour_hist_h, temp_hist_h = heating(distances, tour_hist_c[-1], beta_max)
        
        best_cost.extend([best_cost_c, best_cost_h])
        best_tour.extend([best_tour_c, best_tour_h])
        cost_history.extend(cost_hist_c + cost_hist_h)
        tour_history.extend(tour_hist_c + tour_hist_h)
        temperature_history.extend(temp_hist_c + temp_hist_h)
        
        cycle = 1
        start_sa = time.time()
        
        while cycle <= annealing_cycles:
            best_cost_c, best_tour_c, cost_hist_c, tour_hist_c, temp_hist_c = cooling(distances, tour_history[-1], beta_min)
            best_cost_h, best_tour_h, cost_hist_h, tour_hist_h, temp_hist_h = heating(distances, tour_hist_c[-1], beta_max)
        
            best_cost.extend([best_cost_c, best_cost_h])
            best_tour.extend([best_tour_c, best_tour_h])
            cost_history.extend(cost_hist_c + cost_hist_h)
            tour_history.extend(tour_hist_c + tour_hist_h)
            temperature_history.extend(temp_hist_c + temp_hist_h)
            
            cycle += 1
        
        end_sa = time.time()
        time_sa = end_sa - start_sa
        
        # Calculate success probability
        freq_ground_state = best_cost.count(cost_ground_state)
        prob_gs = freq_ground_state / len(best_cost)    
            
        gs_probability[run_idx] = prob_gs
        execution_times_sa[run_idx] = time_sa
        
        best_cost_tot[run_idx,] = best_cost
        best_tour_tot[run_idx,] = best_tour
        cost_history_tot[run_idx,] = cost_history
        tour_history_tot[run_idx,] = tour_history
        temperature_history_tot[run_idx,] = temperature_history
    
    # Save the consolidated data for this N
    np.savez(f"../data/gs_comparison_tsp_N{N}.npz",
             best_costs=best_cost_tot, 
             best_tours=best_tour_tot, 
             cost_history=cost_history_tot, 
             tour_history=tour_history_tot, 
             temperature_history=temperature_history_tot, 
             execution_times_sa=execution_times_sa,
             gs_probability=gs_probability,
             tour_ground_state=tour_ground_state_tot,
             cost_ground_state=cost_ground_state_tot,
             execution_times_bf=execution_times_bf)