import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter
from itertools import permutations
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
from adjustText import adjust_text

# -----------------------------------------
# || VARIABLE DECLARATION ||
# -----------------------------------------

N = 4  # Number of nodes (cities to visit)
annealing_cycles = 4

# Temperature parameters
beta_min = 0.0005  # Corresponds to T = 2000 (Initial max temperature)
beta_max = 2000    # Corresponds to T = 0.0005 (Final min temperature)

beta_update_interval_c = 50  # Beta update frequency during cooling
beta_update_interval_h = 50  # Beta update frequency during heating

cooling_rate = 1.05  # Temperature decreases by ~5%
heating_rate = 1.05  # Temperature increases by ~5%

# Data saving interval
save_interval = 50

# -----------------------------------------
# || DATA LOADING (Italian Cities) ||
# -----------------------------------------

# Ensure this path matches your repository structure
data = np.load(f'../data/matrici_distanze/matrice_distanze_italia_N{N}.npz', allow_pickle=True)

distances = data['distanze']
distances_df = data['distanze_df']
cities = data['citta']
points = data['punti']

# -----------------------------------------
# || FUNCTION DEFINITIONS ||
# -----------------------------------------

def exact_ground_state(distance_matrix):
    """Finds the absolute minimum cost using Brute Force."""
    visited_tours = []
    costs = []
    
    for perm in permutations(range(1, N)):
        tour = [0] + list(perm) # Fix the first node
        cost = sum(distance_matrix[tour[j], tour[(j + 1) % len(tour)]] for j in range(len(tour)))
        visited_tours.append(tour)
        costs.append(cost)
        
    gs_cost = min(costs)
    gs_index = np.argmin(costs)
    gs_tour = visited_tours[gs_index]
    
    return gs_tour, gs_cost

# Calculate exact ground state before starting
tour_ground_state, cost_ground_state = exact_ground_state(distances)


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
# || MAIN ANNEALING CYCLE ||
# -----------------------------------------

best_cost = []
best_tour = []
cost_history = []
tour_history = []
temperature_history = []

# Cycle 0
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

while cycle <= annealing_cycles:
    best_cost_c, best_tour_c, cost_hist_c, tour_hist_c, temp_hist_c = cooling(distances, tour_history[-1], beta_min)
    best_cost_h, best_tour_h, cost_hist_h, tour_hist_h, temp_hist_h = heating(distances, tour_hist_c[-1], beta_max)

    best_cost.extend([best_cost_c, best_cost_h])
    best_tour.extend([best_tour_c, best_tour_h])
    cost_history.extend(cost_hist_c + cost_hist_h)
    tour_history.extend(tour_hist_c + tour_hist_h)
    temperature_history.extend(temp_hist_c + temp_hist_h)
    
    cycle += 1

# Check if ground state was found
freq_ground_state = best_tour.count(tour_ground_state)
prob_gs = freq_ground_state / len(best_tour)

# -----------------------------------------
# || PLOT 1: COST vs TEMPERATURE ||
# -----------------------------------------

x_axis = np.arange(0, len(cost_history), 1) * save_interval

fig, ax1 = plt.subplots(figsize=(12, 8))

# Primary axis: COST
ax1.plot(x_axis, cost_history, 'g-', label='Cost', linewidth=1)
ax1.set_xlabel("\nIterations", fontsize=12)
ax1.set_ylabel("Cost (km)\n", color='g', fontsize=12)
ax1.tick_params(axis='y', labelcolor='g')

ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.xaxis.get_major_formatter().set_scientific(True)
ax1.xaxis.get_major_formatter().set_powerlimits((-2, 2))

# Secondary axis: TEMPERATURE
ax2 = ax1.twinx()
ax2.plot(x_axis, temperature_history, c='m', linestyle='--', label='Temperature', linewidth=1.75)
ax2.set_ylabel("\nTemperature (T)", color='m', fontsize=12)
ax2.tick_params(axis='y', labelcolor='m')

plt.tight_layout()
plt.savefig(f'../assets/tsp_sa_italy_N{N}_cost_temp.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

# -----------------------------------------
# || PLOT 2: CITIES ON GEOGRAPHIC MAP ||
# -----------------------------------------

# Create geometries from lat/lon (Note: usually it's lon, lat)
geometry = [Point(lon, lat) for lat, lon in points]
gdf = gpd.GeoDataFrame({'city': cities}, geometry=geometry, crs='EPSG:4326')

# Reproject for contextily basemap compatibility
gdf = gdf.to_crs(epsg=3857)

fig_map, ax_map = plt.subplots(figsize=(12, 10))
gdf.plot(ax=ax_map, color='red', markersize=20, edgecolor='black')

texts = []
for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf['city']):
    texts.append(ax_map.text(x, y, label, fontsize=6, ha='right', va='bottom'))

adjust_text(texts, ax=ax_map, only_move={'points': 'y', 'text': 'y'}, expand_text=(1.5, 1.5), force_text=0.6, precision=0.001, lim=200)

ctx.add_basemap(ax_map, source=ctx.providers.CartoDB.PositronNoLabels)

ax_map.set_axis_off()
plt.title(f"TSP - {N} Italian Cities", fontsize=15)
plt.tight_layout()
plt.show()

# -----------------------------------------
# || PLOT 3: EXACT GROUND STATE ROUTE ||
# -----------------------------------------

pos = {i: (geom.x, geom.y) for i, geom in enumerate(gdf.geometry)}

G = nx.Graph()
for i in range(N):
    G.add_node(i)
    for j in range(i + 1, N):
        G.add_edge(i, j, weight=distances[i, j])

fig_gs, ax_gs = plt.subplots(figsize=(12, 12))

nx.draw_networkx_nodes(G, pos, ax=ax_gs, node_color='blue', node_size=50)

path_edges = list(zip(tour_ground_state, tour_ground_state[1:] + [tour_ground_state[0]]))
all_edges = list(G.edges())
transparent_edges = [e for e in all_edges if e not in path_edges and (e[1], e[0]) not in path_edges]

nx.draw_networkx_edges(G, pos, edgelist=transparent_edges, edge_color='gray', width=0.75, alpha=0.2, ax=ax_gs)
nx.draw_networkx_labels(G, pos, ax=ax_gs, font_size=6, font_color='white', font_weight='bold')
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=1, ax=ax_gs)

texts_gs = []
for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf['city']):
    texts_gs.append(ax_gs.text(x, y, label, fontsize=5, ha='right', va='bottom'))

adjust_text(texts_gs, ax=ax_gs, only_move={'points': 'y', 'text': 'y'}, expand_text=(1.5, 1.5), force_text=0.6, precision=0.001, lim=200)

ctx.add_basemap(ax_gs, source=ctx.providers.CartoDB.PositronNoLabels)

ax_gs.set_title(f"\nTSP - {N} Italian Cities\n Exact Ground State Cost = {cost_ground_state:.3f} km", fontsize=12)
ax_gs.axis('off')
ax_gs.set_aspect('equal')

plt.savefig(f'../assets/tsp_ground_state_italy_N{N}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

# -----------------------------------------
# || PLOT 4: ANIMATION (GIF) ||
# -----------------------------------------

fig_gif, ax_gif = plt.subplots(figsize=(12, 12))

def update_gif(frame):
    ax_gif.clear()
    tour = tour_history[frame]

    nx.draw_networkx_nodes(G, pos, ax=ax_gif, node_color='blue', node_size=150)
    
    path_edges_gif = list(zip(tour, tour[1:] + [tour[0]]))
    transparent_edges_gif = [e for e in all_edges if e not in path_edges_gif and (e[1], e[0]) not in path_edges_gif]
    
    nx.draw_networkx_edges(G, pos, edgelist=transparent_edges_gif, edge_color='gray', width=0.75, alpha=0.2, ax=ax_gif)
    nx.draw_networkx_labels(G, pos, ax=ax_gif, font_size=8, font_color='white', font_weight='bold')
    nx.draw_networkx_edges(G, pos, edgelist=path_edges_gif, edge_color='red', width=1, ax=ax_gif)
    
    ctx.add_basemap(ax_gif, source=ctx.providers.CartoDB.PositronNoLabels)
    
    ax_gif.set_title(f"\nTSP - {N} Italian Cities \n Iteration {frame*save_interval} | Cost: {cost_history[frame]:.0f} km | T: {temperature_history[frame]:.4f}", fontsize=14)
    ax_gif.axis('off')
    ax_gif.set_aspect('equal')

frames_to_use = range(0, int(np.log(beta_max/beta_min)/np.log(cooling_rate)), 5)  
ani = FuncAnimation(fig_gif, update_gif, frames=frames_to_use, interval=300)

ani.save(f'../assets/tsp_sa_italy_N{N}.gif', writer=PillowWriter(fps=5))