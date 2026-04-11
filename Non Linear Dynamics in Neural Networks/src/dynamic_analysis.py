import scipy.io
import numpy as np

def sliding_window_connectivity(time_series, window_size, step_size, sigma=None):
    """
    time_series: shape (T, N)
    window_size: number of timepoints per window
    step_size: shift between windows
    sigma: std of Gaussian (default = window_size/3)
    """
    num_time_points, num_regions = time_series.shape
    num_windows = (num_time_points - window_size) // step_size + 1
    
    connectivity_matrices = []
    connectivity_matrices = np.zeros((num_windows, num_regions, num_regions))
    
    # Default sigma
    if sigma is None:
        sigma = window_size / 3
    
    # Create Gaussian window
    x = np.arange(window_size)
    center = (window_size - 1) / 2

    gaussian_window = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    # Normalize (optional but recommended)
    gaussian_window = gaussian_window / np.sum(gaussian_window)
    
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        
        window_ts = time_series[start:end]
        
        # Apply Gaussian taper
        window_ts_weighted = window_ts * gaussian_window[:, None]
        
        # Correlation matrix
        connectivity_matrices[i] = np.corrcoef(window_ts_weighted, rowvar=False) 
        
    return connectivity_matrices

def spectral_partition(connectivity_matrix):
    W = np.abs(connectivity_matrix)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1] #second smallest eigenvector 
    cluster_labels = np.where(fiedler_vector >= 0, 1, 0) #binary partition

    return cluster_labels, fiedler_vector

def spectral_clustering(connectivity_matrices):
    num_windows = connectivity_matrices.shape[0]
    N = connectivity_matrices.shape[1]  

    all_clusters = np.zeros((num_windows, N))
    all_fiedlers = np.zeros((num_windows, N))

    for i in range(num_windows):
        cluster_labels, fiedler_vector = spectral_partition(connectivity_matrices[i])
        all_clusters[i] = cluster_labels
        all_fiedlers[i] = fiedler_vector

    return all_clusters, all_fiedlers

def compute_roi_switches(cluster_matrix):
    return np.sum(np.abs(np.diff(cluster_matrix, axis=0)), axis=0)