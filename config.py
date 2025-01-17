# config.py
WINDOW_SIZE = 10  # Sliding window size for covariance computation
LAMBDA = 0.1      # Regularization parameter for sparsity
BETA = 0.2        # Temporal consistency penalty
RHO = 1.0         # ADMM penalty parameter
MAX_ITER = 100    # Maximum iterations for ADMM
TOL = 1e-4        # Convergence tolerance
