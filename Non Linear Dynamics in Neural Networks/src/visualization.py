import matplotlib.pyplot as plt
import numpy as np
import os

def time_series(atlas_labels, bold_ts, bold_ts_pd):
    roi_indices = [0, 1, 2, 3, 4]
    fig, axes = plt.subplots(nrows = 2, ncols = 1,figsize=(10, 5))
    for roi in roi_indices:
        axes[0].plot(bold_ts[:, roi], label=atlas_labels[roi])
        axes[1].plot(bold_ts_pd[:, roi], label=atlas_labels[roi])
    axes[0].set_xlabel("Time points")
    axes[1].set_xlabel("Time points")
    axes[0].set_ylabel("BOLD signal (standardized)")
    axes[1].set_ylabel("BOLD signal (standardized)")
    axes[0].set_title("ROI-wise BOLD Time Series (Healthy)")
    axes[1].set_title("ROI-wise BOLD Time Series (Parkinsons)")
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/roi_time_series.png", dpi=300)
    print("\nROI plot saved to results/plots/roi_time_series.png")