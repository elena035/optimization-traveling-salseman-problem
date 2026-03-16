import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter

# -----------------------------------------
# || VARIABLE DECLARATION ||
# -----------------------------------------

N = 30 # Number of nodes (cities to visit)
annealing_cycles = 4

# Temperature parameters
beta_min = 0.003 # Corresponds to T = ~333 (High T, initial state)
beta_max = 1000  # Corresponds to T = 0.001 (Low T, final state) 

beta_update_interval_c = 100 # Frequency of beta update during cooling
beta_update_interval_h = 100 # Frequency of beta update during heating

cooling_rate = 1.02 # Decrease temperature by ~2% each update
heating_rate = 1.05 # Increase temperature by ~5% each update

# Data saving parameters
save_interval = 100

# --------------------------------------------------
# || SYNTHETIC MODULAR DISTRIBUTION SETUP ||
# --------------------------------------------------

np.random.seed(42)  # For reproducibility

# Modular parameters
num_clusters = 4
points_per_cluster = N // num_clusters
cluster_std = 4

# Cluster centers
cluster_centers = np.random.rand(num_clusters, 2) * 100 

# Generate nodes (cities)
points = []
for center in cluster_centers:
    cluster_points = np.random.randn(points_per_cluster, 2) * cluster_std + center     
    points.append(cluster_points)

points = np.vstack(points) 

# Add extra points if N is not divisible by num_clusters
while len(points) < N:
    extra = np.random.randn(1, 2) * cluster_std + cluster_centers[0]
    points = np.vstack([points, extra])

# Calculate Euclidean distance matrix
distances = np.zeros((N, N))
for i in range(N):
    for j in range(i + 1, N):
        dist = np.linalg.norm(points[i] - points[j])
        distances[i, j] = dist
        distances[j, i] = dist 

# -----------------------------------------
# || FUNCTION DEFINITIONS ||
# -----------------------------------------

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
        # Swap two random cities
        i, j = random.sample(range(N), 2)
        new_tour = current_tour[:]
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i] 
        
        new_cost = sum(distance_matrix[new_tour[i], new_tour[(i+1)%len(current_tour)]] for i in range(len(new_tour)))
        delta = new_cost - current_cost 

        # Metropolis criterion
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
        i, j = random.sample(range(N), 2)
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
# || MAIN ANNEALING CYCLE ||
# -----------------------------------------

best_cost = []
best_tour = []
cost_history = []
tour_history = []
temperature_history = []

# Cycle 0 (Initialization)
start_tour = list(range(N))
random.shuffle(start_tour)

best_cost_c, best_tour_c, cost_hist_c, tour_hist_c, temp_hist_c = cooling(distances, start_tour, beta_min)
best_cost_h, best_tour_h, cost_hist_h, tour_hist_h, temp_hist_h = heating(distances, tour_hist_c[-1], beta_max)

best_cost.extend([best_cost_c, best_cost_h])
best_tour.extend([best_tour_c, best_tour_h])
cost_history.extend(cost_hist_c + cost_hist_h)
tour_history.extend(tour_hist_c + tour_hist_h)
temperature_history.extend(temp_hist_c + temp_hist_h)

cycle = 1

while cycle <= annealing_cycles:
    best_cost_c, best_tour_c, cost_hist_c, tour_hist_c, temp_hist_c = cooling(distances, tour_history[-1], beta_min)
    best_cost_h, best_tour_h, cost_hist_h, tour_hist_h, temp_hist_h = heating(distances, tour_hist_c[-1], beta_max)

    best_cost.extend([best_cost_c, best_cost_h])
    best_tour.extend([best_tour_c, best_tour_h])
    cost_history.extend(cost_hist_c + cost_hist_h)
    tour_history.extend(tour_hist_c + tour_hist_h)
    temperature_history.extend(temp_hist_c + temp_hist_h)
    
    cycle += 1

# -----------------------------------------
# || PLOTTING AND EXPORTS ||
# -----------------------------------------

x_axis = np.arange(0, len(cost_history), 1) * 100

# 1. Static Plot
fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.plot(x_axis, cost_history, '#3c4043', label='Cost', linewidth=1.75)
ax1.set_xlabel("\nIterations", fontsize=12)
ax1.set_ylabel("Cost\n", color='#3c4043', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#3c4043')
ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.xaxis.get_major_formatter().set_scientific(True)
ax1.xaxis.get_major_formatter().set_powerlimits((-2, 2))

ax2 = ax1.twinx()
ax2.plot(x_axis, temperature_history, c='#ad3e79', linestyle='--', label='Temperature', linewidth=2)
ax2.set_ylabel("\nTemperature (T)", color='#ad3e79', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#ad3e79')

plt.tight_layout()
plt.savefig(f'../assets/tsp_sa_{annealing_cycles}cycles_1run_N{N}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

# 2. Dynamic Plot GIF (Cost vs Temp)
frames_to_use = list(range(0, int(np.log(beta_max/beta_min)/np.log(cooling_rate)), 5))

cost_hist_reduced = [cost_history[i] for i in frames_to_use]
temp_hist_reduced = [temperature_history[i] for i in frames_to_use]
x_axis_reduced = [i * 100 for i in frames_to_use]  

fig_gif, ax1_gif = plt.subplots(figsize=(12, 8))
ax2_gif = ax1_gif.twinx()

line1, = ax1_gif.plot([], [], '#3c4043', label='Cost', linewidth=1.75)
line2, = ax2_gif.plot([], [], c='#ad3e79', linestyle='--', label='Temperature', linewidth=2)

ax1_gif.set_xlim(x_axis_reduced[0], x_axis_reduced[-1])
ax1_gif.set_ylim(min(cost_hist_reduced) * 0.95, max(cost_hist_reduced) * 1.05)
ax2_gif.set_ylim(min(temp_hist_reduced) * 0.95, max(temp_hist_reduced) * 1.05)

ax1_gif.set_xlabel("\nIterations", fontsize=12)
ax1_gif.set_ylabel("Cost\n", color='#3c4043', fontsize=12)
ax2_gif.set_ylabel("\nTemperature (T)", color='#ad3e79', fontsize=12)
ax1_gif.tick_params(axis='y', labelcolor='#3c4043')
ax2_gif.tick_params(axis='y', labelcolor='#ad3e79')

ax1_gif.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1_gif.xaxis.get_major_formatter().set_scientific(True)
ax1_gif.xaxis.get_major_formatter().set_powerlimits((-2, 2))

plt.tight_layout()

def update_line_chart(frame_idx):
    i = frame_idx + 1 
    line1.set_data(x_axis_reduced[:i], cost_hist_reduced[:i])
    line2.set_data(x_axis_reduced[:i], temp_hist_reduced[:i])
    return line1, line2

ani1 = FuncAnimation(fig_gif, update_line_chart, frames=len(frames_to_use), interval=300, blit=True)
ani1.save('../assets/tsp_sa_cost_temperature.gif', writer=PillowWriter(fps=5))

# 3. Graph Evolution GIF
if N <= 50:
    G = nx.Graph()
    n_nodes = len(distances)
    for i in range(n_nodes):
        G.add_node(i)
        for j in range(i + 1, n_nodes):
            G.add_edge(i, j, weight=distances[i, j]) 

    pos = {i: points[i] for i in range(N)}
    fig_graph, ax_graph = plt.subplots(figsize=(8, 8))
    
    def update_graph(frame):
        ax_graph.clear()
        tour = tour_history[frame]
        
        nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color='#6e3855', node_size=150)
        
        path_edges = list(zip(tour, tour[1:])) + [(tour[-1], tour[0])]
        all_edges = list(G.edges())
        transparent_edges = [e for e in all_edges if e not in path_edges and (e[1], e[0]) not in path_edges]
        
        nx.draw_networkx_edges(G, pos, edgelist=transparent_edges, edge_color='#595959', width=0.5, alpha=0.2, ax=ax_graph)
        nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=7, font_color='white', font_weight='bold')
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='#ad3e79', width=1.5, ax=ax_graph)
        
        ax_graph.set_title(f"\n Iteration {frame*100} | Cost: {cost_history[frame]:.2f} | T: {temperature_history[frame]:.3f}", fontsize=14)
        ax_graph.axis('off')

    ani2 = FuncAnimation(fig_graph, update_graph, frames=frames_to_use, interval=300)
    ani2.save('../assets/tsp_sa_single_run_graph.gif', writer=PillowWriter(fps=5))
    
# Save data array
np.savez('../data/data_gif_presentation.npz',
         best_costs=best_cost, 
         best_tours=best_tour, 
         cost_history=cost_history, 
         tour_history=tour_history, 
         temperature_history=temperature_history)