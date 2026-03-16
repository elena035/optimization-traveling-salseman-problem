import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import permutations

# ------------------------
# || GENERAL PARAMETERS ||
# ------------------------

min_distance = 0
max_distance = 100

num_clusters = 4
cluster_std = 4

N_values = [4, 6, 8, 10]  # Values of N (nodes) to consider

distance_matrices = {}
point_coords = {}
gs_tours = {}
gs_costs = {}

# -----------------------------
# || GENERATE GRAPHS & TOURS ||
# -----------------------------

for N in N_values:
    points_per_cluster = N // num_clusters

    # Cluster centers
    cluster_centers = np.random.rand(num_clusters, 2) * 100

    # Generate points
    points = []
    for center in cluster_centers:
        cluster_points = np.random.randn(points_per_cluster, 2) * cluster_std + center
        points.append(cluster_points)

    points = np.vstack(points)

    # Add extra points if N is not perfectly divisible
    while len(points) < N:
        extra = np.random.randn(1, 2) * cluster_std + cluster_centers[0]
        points = np.vstack([points, extra])

    # Calculate distance matrix
    distances = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(points[i] - points[j])
            distances[i, j] = dist
            distances[j, i] = dist

    distances = np.clip(distances, min_distance, max_distance)

    # Save for future use
    distance_matrices[N] = distances
    point_coords[N] = points

    # --------------------------
    # || CALCULATE EXACT G.S. ||
    # --------------------------
    # Using brute force (permutations) to find the absolute minimum cost
    
    visited_tours = []
    costs = []

    # We fix the starting node (0) to avoid redundant reversed paths
    for perm in permutations(range(1, N)):
        tour = [0] + list(perm)
        cost = sum(distances[tour[j], tour[(j + 1) % N]] for j in range(N))
        visited_tours.append(tour)
        costs.append(cost)

    best_idx = np.argmin(costs)
    gs_tours[N] = visited_tours[best_idx]
    gs_costs[N] = costs[best_idx]


# -----------------------------
# || PLOT ALL GRAPHS         ||
# -----------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes = axes.flatten()

for idx, N in enumerate(N_values):
    ax = axes[idx]
    distances = distance_matrices[N]
    points = point_coords[N]
    tour = gs_tours[N]

    # Create Graph
    G = nx.Graph()
    for i in range(N):
        G.add_node(i)
        for j in range(i + 1, N):
            G.add_edge(i, j, weight=distances[i, j])

    pos = {i: points[i] for i in range(N)}
    
    # Identify edges belonging to the optimal tour
    path_edges = list(zip(tour, tour[1:] + [tour[0]]))
    all_edges = list(G.edges())
    transparent_edges = [e for e in all_edges if e not in path_edges and (e[1], e[0]) not in path_edges]

    # Draw nodes and background edges
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#6e3855', node_size=150)
    nx.draw_networkx_edges(G, pos, edgelist=transparent_edges, edge_color='#595959', width=0.75, alpha=0.4, ax=ax)
    
    # Draw labels and optimal tour
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color='white', font_weight='bold')
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='#ad3e79', width=1.5, ax=ax)

    ax.set_title(f"TSP - {N} nodes\nExact Ground State Cost: {gs_costs[N]:.2f}", fontsize=14)

plt.tight_layout(pad=5.0)
plt.subplots_adjust(hspace=0.2, wspace=0.2) 

# Save the figure to the assets folder
plt.savefig('../assets/synthetic_exact_gs_representations.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

plt.show()