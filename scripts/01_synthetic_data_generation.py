import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math

N = 8

# Modular parameters for the clusters
num_clusters = int(np.sqrt(N))
points_per_cluster = N // num_clusters
cluster_std = np.sqrt(N) / 20  # Spread of the points within each cluster

# Generate cluster centers randomly in space
# The coordinates are distributed within the square [0, sqrt(N)]
cluster_centers = np.random.rand(num_clusters, 2) * np.sqrt(N)

# Generate the nodes (points)
points = []
for center in cluster_centers:
    # Sample coordinates from a standard Gaussian, scale by cluster_std, and shift by the center
    cluster_points = np.random.randn(points_per_cluster, 2) * cluster_std + center     
    points.append(cluster_points)

# Convert the list of 2D arrays into an (N, 2) matrix
points = np.vstack(points)

# If N is not perfectly divisible by num_clusters, add the missing points to the first cluster
while len(points) < N:
    extra = np.random.randn(1, 2) * cluster_std + cluster_centers[0]
    points = np.vstack([points, extra])

# Compute the Euclidean distance matrix (symmetric with zero diagonal)
distances = np.zeros((N, N))
for i in range(N):
    for j in range(i + 1, N):
        dist = np.linalg.norm(points[i] - points[j])
        distances[i, j] = dist
        distances[j, i] = dist

# === CREATE THE GRAPH ===
G = nx.Graph()
for i in range(N):
    G.add_node(i)
    for j in range(i + 1, N):
        G.add_edge(i, j, weight=distances[i, j]) 

# Map the real coordinates to the graph nodes
pos = {i: points[i] for i in range(N)}

# === PLOT THE DISTRIBUTION ===
fig, ax = plt.subplots(figsize=(8, 8))

x_values = [coord[0] for coord in pos.values()]
y_values = [coord[1] for coord in pos.values()]

# Draw the nodes
nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#6e3855', node_size=50, hide_ticks=False)

# Draw transparent scatter points to force matplotlib to show the numeric axes
ax.scatter(x_values, y_values, alpha=0)

# Plot formatting
ax.set_title(f"N = {N}", fontsize=14)
ax.grid(True, which='both', color='gray', alpha=0.2)
ax.set_axis_on()

# Set axis limits dynamically based on N
ax.set_xlim(-0.05, math.ceil(np.sqrt(N)) + 0.05)
ax.set_ylim(-0.05, math.ceil(np.sqrt(N)) + 0.05)

ax.set_xlabel('X coordinate', fontsize=12)
ax.set_ylabel('Y coordinate', fontsize=12)

# Save the figure (Update the path as needed before uncommenting)
# plt.savefig(f'../assets/synthetic_distribution_N{N}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

plt.show()