import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

def network(correlation_matrix, correlation_matrix_parkinsons, atlas_labels):
    threshold = 0.5
    adjacency_matrix_healthy = correlation_matrix.copy()
    adjacency_matrix_healthy[np.abs(adjacency_matrix_healthy) < threshold] = 0

    adjacency_matrix_parkinsons = correlation_matrix_parkinsons.copy()
    adjacency_matrix_parkinsons[np.abs(adjacency_matrix_parkinsons) < threshold] = 0

    mapping = {i: atlas_labels[i] for i in range(len(atlas_labels))}

    G_healthy = nx.from_numpy_array(adjacency_matrix_healthy)
    G_healthy = nx.relabel_nodes(G_healthy, mapping)

    G_parkinsons = nx.from_numpy_array(adjacency_matrix_parkinsons)
    G_parkinsons = nx.relabel_nodes(G_parkinsons, mapping)

    print("\nHealthy Subject Graph Analysis:")
    print("Number of nodes:", G_healthy.number_of_nodes())
    print("Number of edges:", G_healthy.number_of_edges())

    degree_healthy = nx.degree_centrality(G_healthy)
    clustering_healthy = nx.clustering(G_healthy, weight = "weight")
    top_hubs_healthy = sorted(degree_healthy.items(), key = lambda  x: x[1], reverse=True)[:5]
    print("\nTop 5 hubs (by degree centrality):")
    for i in top_hubs_healthy:
        print(i)

    print("\nParkinsons Subject Graph Analysis:")
    print("Number of nodes:", G_parkinsons.number_of_nodes())
    print("Number of edges:", G_parkinsons.number_of_edges())

    degree_parkinsons = nx.degree_centrality(G_parkinsons)
    clustering_parkinsons = nx.clustering(G_parkinsons, weight = "weight")
    top_hubs_parkinsons = sorted(degree_parkinsons.items(), key = lambda  x: x[1], reverse=True)[:5]
    print("\nTop 5 hubs (by degree centrality):")
    for i in top_hubs_parkinsons:
        print(i)


    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(20, 10))
    pos_h = nx.spring_layout(G_healthy, seed=42)
    pos_p = nx.spring_layout(G_parkinsons, seed=42)
    nx.draw(
        G_healthy,
        ax = axes[0], 
        pos = pos_h, 
        with_labels=True, 
        node_size=500, 
        node_color="skyblue", 
        font_size=10, 
        font_weight="bold", 
        edge_color="gray"
    )
    axes[0].set_title("Functional Connectivity Network (Healthy) (Thresholded at {})".format(threshold))
    nx.draw(
        G_parkinsons,
        ax = axes[1], 
        pos = pos_p, 
        with_labels=True, 
        node_size=500, 
        node_color="green", 
        font_size=10, 
        font_weight="bold", 
        edge_color="gray"
    )
    axes[1].set_title("Functional Connectivity Network (Parkinsons) (Thresholded at {})".format(threshold))
    os.makedirs("results/graphs", exist_ok=True)
    plt.savefig("results/graphs/brain_network_graph.png", dpi=300)
    plt.tight_layout()
    plt.show()