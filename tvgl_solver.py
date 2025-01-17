# tvgl_solver.py
import numpy as np
from utils import soft_threshold, log_det, check_convergence

def solve_tvgl(covariances, lambda_, beta, rho, max_iter, tol):
    """
    Solve the TVGL optimization problem using ADMM.
    """
    T = len(covariances)
    p = covariances[0].shape[0]
    
    # Initialize variables
    Theta = [np.eye(p) for _ in range(T)]
    Z = [np.eye(p) for _ in range(T)]
    U = [np.zeros((p, p)) for _ in range(T)]
    
    for iteration in range(max_iter):
        # Update Theta
        for t in range(T):
            Theta[t] = solve_theta(covariances[t], Z[t], U[t], rho)
        
        # Update Z
        for t in range(T):
            if t > 0:
                diff = Theta[t] - Theta[t-1]
                temporal_penalty = beta * diff
                Z[t] = soft_threshold(Theta[t] + U[t] - temporal_penalty, lambda_ / rho)
            else:
                Z[t] = soft_threshold(Theta[t] + U[t], lambda_ / rho)
        
        # Update U
        for t in range(T):
            U[t] += Theta[t] - Z[t]
        
        # Check convergence
        if check_convergence(Theta, Z, tol):
            break
    
    return Theta

def solve_theta(S, Z, U, rho):
    """
    Solve the subproblem for Theta (precision matrix update).
    """
    return np.linalg.inv(S + rho * (Z - U))  # Simplified for demonstration

