import pyreadr
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import random
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from shapely.geometry import Point
import contextily as ctx

# -----------------------------------------
# || VARIABLE DECLARATION ||
# -----------------------------------------

annealing_cycles = 4

# Temperature parameters
beta_min = 0.0002 # Corresponds to T = 5000 (Initial max temperature)
beta_max = 5000   # Corresponds to T = 0.0002 (Final min temperature) 

beta_update_interval_c = 300 # Frequency of beta update during cooling
beta_update_interval_h = 300 # Frequency of beta update during heating

cooling_rate = 1.01 # Temperature decreases by ~1%
heating_rate = 1.01 # Temperature increases by ~1%

# Data saving interval
save_interval = 300

# -----------------------------------------
# || LOAD AND PREPARE DATA (.rds) ||
# -----------------------------------------

# Load the .rds file containing worldwide cities
# Update path to match your repository structure
result = pyreadr.read_r('../data/citta_reali/cities.rds')
df = next(iter(result.values()))

# Extract coordinates
points = df[['lat', 'long']].to_numpy()
N_total = len(points)

# Calculate geodesic distance matrix in km
distances_total = np.zeros((N_total, N_total))
for i in range(N_total):
    for j in range(i + 1, N_total):
        dist = geodesic(points[i], points[j]).km
        distances_total[i, j] = dist
        distances_total[j, i] = dist
        
# Create readable DataFrame with city names
cities = df['name'].values
distances_df_total = pd.DataFrame(distances_total, index=cities, columns=cities)

# -----------------------------------------
# || RANDOM SUBSET SELECTION (500 Cities) ||
# -----------------------------------------

N_cut = 500 # Number of cities to actually use for the TSP

# Randomly select 500 indices without replacement
selected_indices = np.random.choice(N_total, size=N_cut, replace=False)

# Extract the 500x500 submatrix using np.ix_
distances = distances_total[np.ix_(selected_indices, selected_indices)]
cities_cut = cities[selected_indices]
distances_df_cut = pd.DataFrame(distances, index=cities_cut, columns=cities_cut)

# Save the subset distance matrix for future runs (e.g., T_max tests)
np.savez('../data/distance_matrix_ww_500.npz',
         distance_matrix = distances, distance_matrix_df = distances_df_cut)

# -----------------------------------------
# || GEOGRAPHIC PLOT OF SELECTED CITIES ||
# -----------------------------------------

# Create geometries from lat/lon for GeoPandas (remember (lon, lat) order)
geometry = [Point(xy) for xy in zip(df.iloc[selected_indices]['long'], df.iloc[selected_indices]['lat'])]
gdf = gpd.GeoDataFrame(df.iloc[selected_indices], geometry=geometry, crs='EPSG:4326') 

# Project to Web Mercator for basemap compatibility
gdf = gdf.to_crs(epsg=3857)

fig_map, ax_map = plt.subplots(figsize=(16, 14))

# Plot the points
gdf.plot(ax=ax_map, color='red', markersize=3, edgecolor='black', linewidth=0.2)

# Add OpenStreetMap basemap
ctx.add_basemap(ax_map, source=ctx.providers.OpenStreetMap.Mapnik)

ax_map.set_axis_off()
ax_map.set_xlim(gdf.total_bounds[0] - 1e5, gdf.total_bounds[2] + 1e5)
ax_map.set_ylim(gdf.total_bounds[1] - 1e5, gdf.total_bounds[3] + 1e5)

plt.title(f"\n{N_cut} Random Cities Worldwide\n", fontsize=15)
plt.tight_layout()

plt.savefig('../assets/map_ww_500_cities.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

# -----------------------------------------
# || CORE ANNEALING FUNCTIONS ||
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
        i, j = random.sample(range(N_cut), 2)
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
        i, j = random.sample(range(N_cut), 2)
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
start_tour = list(range(N_cut))
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
# || PLOTTING RESULTS (Cost vs Temp) ||
# -----------------------------------------

x_axis = np.arange(0, len(cost_history), 1) * save_interval

fig_sa, ax1 = plt.subplots(figsize=(12, 8))

# Primary axis: COST
ax1.plot(x_axis, cost_history, 'g-', label='Cost', linewidth=1.75)
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
plt.savefig(f'../assets/tsp_sa_{annealing_cycles}cycles_ww_{N_cut}cities.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()
