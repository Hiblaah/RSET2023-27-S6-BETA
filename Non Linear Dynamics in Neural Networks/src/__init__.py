import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import os

# ==========================================
# 1️⃣ Animated Connectivity Heatmap
# ==========================================

def animate_connectivity(connectivity_matrices, title, save_path):
    fig, ax = plt.subplots(figsize=(6,6))

    def update(frame):
        ax.clear()
        im = ax.imshow(connectivity_matrices[frame], 
                       vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"{title} - Window {frame+1}")
        return [im]

    ani = FuncAnimation(fig, update, frames=len(connectivity_matrices))
    
    os.makedirs("results/animations", exist_ok=True)
    ani.save(save_path, writer="pillow", fps=2)
    plt.close()
    print(f"Saved animation to {save_path}")


# ==========================================
# 2️⃣ Animated Network Graph (Cluster Colored)
# ==========================================

def animate_network(connectivity_matrices, cluster_matrix, atlas_labels, title, save_path, threshold=0.5):

    fig, ax = plt.subplots(figsize=(8,8))

    def update(frame):
        ax.clear()

        corr = connectivity_matrices[frame].copy()
        corr[np.abs(corr) < threshold] = 0

        G = nx.from_numpy_array(corr)

        mapping = {i: atlas_labels[i] for i in range(len(atlas_labels))}
        G = nx.relabel_nodes(G, mapping)

        pos = nx.spring_layout(G, seed=42)

        node_colors = cluster_matrix[frame]

        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=False,
            node_color=node_colors,
            cmap=plt.cm.Set1,
            node_size=300,
            edge_color="gray"
        )

        ax.set_title(f"{title} - Window {frame+1}")

    ani = FuncAnimation(fig, update, frames=len(connectivity_matrices))

    os.makedirs("results/animations", exist_ok=True)
    ani.save(save_path, writer="pillow", fps=2)
    plt.close()
    print(f"Saved animation to {save_path}")


# ==========================================
# 3️⃣ ROI Switching Bar Plot
# ==========================================

def plot_roi_switches(switches_healthy, switches_pd, atlas_labels):

    x = np.arange(len(atlas_labels))

    plt.figure(figsize=(12,5))
    plt.bar(x - 0.2, switches_healthy, width=0.4, label="Healthy")
    plt.bar(x + 0.2, switches_pd, width=0.4, label="Parkinsons")

    plt.xticks(x, atlas_labels, rotation=90)
    plt.ylabel("Number of Community Switches")
    plt.title("ROI Flexibility Comparison")
    plt.legend()
    plt.tight_layout()

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/roi_switching_comparison.png", dpi=300)
    plt.show()

    print("ROI switching comparison saved.")