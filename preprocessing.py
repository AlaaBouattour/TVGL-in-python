# preprocessing.py
import numpy as np
import pandas as pd

def compute_sliding_window_covariance(data, window_size):
    """
    Compute empirical covariance matrices using a sliding window approach.
    """
    n_rows, n_cols = data.shape
    covariances = []
    
    for t in range(window_size, n_rows - window_size):
        window = data.iloc[t - window_size:t + window_size + 1]
        cov = np.cov(window.values, rowvar=False)
        covariances.append(cov)
    
    return covariances

def standardize_data(data):
    """
    Standardize the data to have zero mean and unit variance.
    """
    return (data - data.mean()) / data.std()

def preprocess(data, window_size):
    """
    Full preprocessing pipeline: standardize data and compute covariance matrices.
    """
    standardized_data = standardize_data(data)
    covariances = compute_sliding_window_covariance(standardized_data, window_size)
    return covariances
