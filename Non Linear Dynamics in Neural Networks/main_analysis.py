from src.network import network
from src.visualization import time_series
from src.load_data import load_fmri, preprocess_fmri, load_atlas
from src.extract_bold import extract_bold_timeseries
from src.connectivity import compute_connectivity_matrix
from src.dynamic_analysis import (
    sliding_window_connectivity,
    spectral_clustering,
    compute_roi_switches
)
from src.dynamic_visualization import (
    animate_connectivity,
    animate_network,
    plot_roi_switches
)

from nilearn.image import mean_img
from nilearn.plotting import plot_epi, plot_matrix, show
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# ======================================================
# STATIC VISUALIZATION FUNCTIONS
# ======================================================

def epi_images():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_epi(mean_img(fmri_clean), axes=axes[0], title="Mean fMRI Image (Healthy)")
    plot_epi(mean_img(fmri_clean_pd), axes=axes[1], title="Mean fMRI Image (Parkinsons)")
    show()


def connectivity_visualization():
    global correlation_matrix_healthy
    global correlation_matrix_parkinsons

    correlation_matrix_healthy = compute_connectivity_matrix(bold_ts)
    correlation_matrix_parkinsons = compute_connectivity_matrix(bold_ts_pd)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_matrix(
        correlation_matrix_healthy,
        labels=pd_regions,
        vmax=0.8,
        vmin=-0.8,
        colorbar=True,
        title="Functional Connectivity (Healthy)",
        axes=axes[0]
    )

    plot_matrix(
        correlation_matrix_parkinsons,
        labels=pd_regions,
        vmax=0.8,
        vmin=-0.8,
        colorbar=True,
        title="Functional Connectivity (Parkinsons)",
        axes=axes[1]
    )

    show()


def network_visualization():
    if 'correlation_matrix_healthy' in globals():
        network(correlation_matrix_healthy, correlation_matrix_parkinsons, pd_regions)
    else:
        print("Please run Connectivity Visualization first.")


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    pd_regions = [
        "Hippocampus_L", "Hippocampus_R",
        "Amygdala_L", "Amygdala_R",
        "Caudate_L", "Caudate_R",
        "Putamen_L", "Putamen_R",
        "Thalamus_L", "Thalamus_R",
        "Olfactory_L", "Olfactory_R",
        "Pallidum_L", "Pallidum_R",
        "Precentral_L", "Precentral_R",
        "Postcentral_L", "Postcentral_R",
        "Cingulum_Ant_L", "Cingulum_Ant_R",
        "Frontal_Mid_L", "Frontal_Mid_R",
        "Supp_Motor_Area_L", "Supp_Motor_Area_R"
    ]

    # --------------------------------------------------
    # Load Single Healthy + PD (for static view)
    # --------------------------------------------------

    fmri_path = "dataset/taowu/sub-control032057/func/sub-control032057_task-resting_bold.nii"
    fmri_path_pd = "dataset/taowu/sub-patient032077/func/sub-patient032077_task-resting_bold.nii"

    fmri_img = load_fmri(fmri_path)
    fmri_clean = preprocess_fmri(fmri_img)

    fmri_img_pd = load_fmri(fmri_path_pd)
    fmri_clean_pd = preprocess_fmri(fmri_img_pd)

    atlas_maps, atlas_labels = load_atlas()
    pd_indices = [atlas_labels.index(r) for r in pd_regions if r in atlas_labels]

    bold_ts = extract_bold_timeseries(fmri_clean, atlas_maps)[:, pd_indices]
    bold_ts_pd = extract_bold_timeseries(fmri_clean_pd, atlas_maps)[:, pd_indices]

    time_series(pd_regions, bold_ts, bold_ts_pd)

    print("\nNonlinear Dynamics in Neural Networks\n")

    while True:

        print("\n1. Static Analysis\n2. Dynamic Analysis\n3. Exit")
        ch1 = int(input("Enter your choice: "))

        # =========================
        # STATIC
        # =========================
        if ch1 == 1:
            connectivity_visualization()
            network_visualization()

        # =========================
        # DYNAMIC
        # =========================
        elif ch1 == 2:

            healthy_files = sorted(glob("dataset/taowu/sub-control*/func/*_task-resting_bold.nii"))
            parkinsons_files = sorted(glob("dataset/taowu/sub-patient*/func/*_task-resting_bold.nii"))

            if len(healthy_files) == 0 or len(parkinsons_files) == 0:
                print("No subject files found.")
                continue

            print(f"\nFound {len(healthy_files)} healthy subjects")
            print(f"Found {len(parkinsons_files)} Parkinson's subjects")

            healthy_roi_switches = []
            parkinsons_roi_switches = []

            # -------- Healthy --------
            for file in healthy_files:
                print(file)
                fmri_img = load_fmri(file)
                fmri_clean = preprocess_fmri(fmri_img)
                bold_sub = extract_bold_timeseries(fmri_clean, atlas_maps)[:, pd_indices]

                conn = sliding_window_connectivity(bold_sub, 20, 1)
                clusters, _ = spectral_clustering(conn)
                switches = compute_roi_switches(clusters)

                healthy_roi_switches.append(switches)

            # -------- Parkinson --------
            for file in parkinsons_files:
                print(file)
                fmri_img = load_fmri(file)
                fmri_clean = preprocess_fmri(fmri_img)
                bold_sub = extract_bold_timeseries(fmri_clean, atlas_maps)[:, pd_indices]

                conn = sliding_window_connectivity(bold_sub, 20, 1)
                clusters, _ = spectral_clustering(conn)
                switches = compute_roi_switches(clusters)

                parkinsons_roi_switches.append(switches)

            avg_healthy = np.mean(healthy_roi_switches, axis=0)
            avg_parkinsons = np.mean(parkinsons_roi_switches, axis=0)

            print("\nAverage ROI Switches (Healthy)")
            for r, v in zip(pd_regions, avg_healthy):
                print(f"{r}: {v:.2f}")

            print("\nAverage ROI Switches (Parkinsons)")
            for r, v in zip(pd_regions, avg_parkinsons):
                print(f"{r}: {v:.2f}")

            # ========================================
            # ANIMATION (Representative Subjects)
            # ========================================

            print("\nGenerating animations...")

            # Healthy representative
            fmri_img = load_fmri(healthy_files[0])
            fmri_clean = preprocess_fmri(fmri_img)
            bold_rep_h = extract_bold_timeseries(fmri_clean, atlas_maps)[:, pd_indices]

            conn_h = sliding_window_connectivity(bold_rep_h, 20, 1)
            clusters_h, _ = spectral_clustering(conn_h)
            switches_h = compute_roi_switches(clusters_h)

            # Parkinson representative
            fmri_img = load_fmri(parkinsons_files[0])
            fmri_clean = preprocess_fmri(fmri_img)
            bold_rep_p = extract_bold_timeseries(fmri_clean, atlas_maps)[:, pd_indices]

            conn_p = sliding_window_connectivity(bold_rep_p, 20, 1)
            clusters_p, _ = spectral_clustering(conn_p)
            switches_p = compute_roi_switches(clusters_p)

            animate_connectivity(conn_h, "Healthy Connectivity", "results/animations/healthy_heatmap.gif")
            animate_connectivity(conn_p, "Parkinsons Connectivity", "results/animations/pd_heatmap.gif")

            animate_network(conn_h, clusters_h, pd_regions, "Healthy Network", "results/animations/healthy_network.gif")
            animate_network(conn_p, clusters_p, pd_regions, "Parkinsons Network", "results/animations/pd_network.gif")

            plot_roi_switches(switches_h, switches_p, pd_regions)

        # =========================
        # EXIT
        # =========================
        elif ch1 == 3:
            print("Exiting...")
            break

        else:
            print("Invalid choice.")