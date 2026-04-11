import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import os

# ==========================================
# 1️⃣ Animated Connectivity Heatmap + Frame Images
# ==========================================

def animate_connectivity(connectivity_matrices, title, save_path):

    os.makedirs("results/animations", exist_ok=True)
    os.makedirs("results/frames/heatmaps", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6,6))

    def update(frame):

        ax.clear()

        matrix = connectivity_matrices[frame]

        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"{title} - Window {frame+1}")

        # Save individual frame image
        frame_path = f"results/frames/heatmaps/window_{frame+1:03d}.png"
        plt.savefig(frame_path, dpi=300)

        return [im]

    ani = FuncAnimation(fig, update, frames=len(connectivity_matrices))

    ani.save(save_path, writer="pillow", fps=2)

    plt.close()

    print(f"Saved animation to {save_path}")
    print("Saved individual heatmap frames to results/frames/heatmaps/")


# ==========================================
# 2️⃣ Animated Network Graph (WITH ROI LABELS)
# ==========================================

def animate_network(connectivity_matrices, cluster_matrix, atlas_labels, title, save_path, threshold=0.5):

    os.makedirs("results/animations", exist_ok=True)
    os.makedirs("results/frames/networks", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9,9))

    # Precompute node layout once so nodes don't jump around
    base_corr = connectivity_matrices[0].copy()
    base_corr[np.abs(base_corr) < threshold] = 0
    base_graph = nx.from_numpy_array(base_corr)
    pos = nx.spring_layout(base_graph, seed=42)

    def update(frame):

        ax.clear()

        corr = connectivity_matrices[frame].copy()
        corr[np.abs(corr) < threshold] = 0

        G = nx.from_numpy_array(corr)

        mapping = {i: atlas_labels[i] for i in range(len(atlas_labels))}
        G = nx.relabel_nodes(G, mapping)

        node_colors = cluster_matrix[frame]

        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,          # <-- SHOW ROI LABELS
            node_color=node_colors,
            cmap=plt.cm.Set1,
            node_size=500,
            font_size=7,
            font_weight="bold",
            edge_color="gray"
        )

        ax.set_title(f"{title} - Window {frame+1}")

        # Save frame image
        frame_path = f"results/frames/networks/window_{frame+1:03d}.png"
        plt.savefig(frame_path, dpi=300)

    ani = FuncAnimation(fig, update, frames=len(connectivity_matrices))

    ani.save(save_path, writer="pillow", fps=2)

    plt.close()

    print(f"Saved animation to {save_path}")
    print("Saved individual network frames to results/frames/networks/")


# ==========================================
# 3️⃣ ROI Switching Bar Plot
# ==========================================

def plot_roi_switches(switches_healthy, switches_pd, atlas_labels):

    os.makedirs("results/plots", exist_ok=True)

    x = np.arange(len(atlas_labels))

    plt.figure(figsize=(12,5))

    plt.bar(x - 0.2, switches_healthy, width=0.4, label="Healthy")
    plt.bar(x + 0.2, switches_pd, width=0.4, label="Parkinsons")

    plt.xticks(x, atlas_labels, rotation=90)

    plt.ylabel("Number of Community Switches")
    plt.title("ROI Flexibility Comparison")

    plt.legend()
    plt.tight_layout()

    save_path = "results/plots/roi_switching_comparison.png"

    plt.savefig(save_path, dpi=300)

    plt.show()

    print(f"ROI switching comparison saved to {save_path}")