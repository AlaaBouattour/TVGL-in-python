# utils.py
import numpy as np

def soft_threshold(X, threshold):
    """
    Apply soft-thresholding to the matrix X with a given threshold.
    """
    return np.sign(X) * np.maximum(np.abs(X) - threshold, 0)

def log_det(matrix):
    """
    Compute the log-determinant of a positive definite matrix.
    """
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError("Matrix is not positive definite.")
    return logdet

def check_convergence(Theta, Z, tol):
    """
    Check convergence by comparing Theta and Z updates.
    """
    max_diff = max(np.linalg.norm(Theta[t] - Z[t], ord='fro') for t in range(len(Theta)))
    return max_diff < tol
