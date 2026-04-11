from nilearn.plotting import plot_epi, plot_roi, show
from nilearn.connectome import ConnectivityMeasure
from nilearn.plotting import plot_matrix, plot_connectome, show, find_parcellation_cut_coords, find_xyz_cut_coords
import numpy as np

def compute_connectivity_matrix(bold_ts):
    #how synchronized are the BOLD signals between different ROIs
    correlation_measure = ConnectivityMeasure(kind = 'correlation')
    correlation_matrix = correlation_measure.fit_transform([bold_ts])[0]
    np.fill_diagonal(correlation_matrix, 0)
    #print("\nCorrelation matrix shape:", correlation_matrix.shape)
    return correlation_matrix