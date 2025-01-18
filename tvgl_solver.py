import numpy as np
import cvxpy as cp

def solve_time_varying_graphical_lasso(S, n_list, lambda_, beta, penalty_function, p, T):
    """
    Solves the time-varying graphical lasso problem with a user-specified penalty function.
    
    Parameters:
    - S: List of empirical covariance matrices (one for each time step).
    - n_list: List of sample sizes (number of observations) for each time step.
    - lambda_: Regularization parameter for sparsity.
    - beta: Regularization parameter for smoothness.
    - penalty_function: A callable function implementing one of the ψ(X) penalties.
    - p: Size of the covariance matrix (number of variables).
    - T: Number of time steps.
    
    Returns:
    - List of precision matrices (Theta) for each time step.
    """
    # Define variables
    Thetas = [cp.Variable((p, p), symmetric=True) for _ in range(T)]
    
    # Constraints
    constraints = [Theta >> 0 for Theta in Thetas]  # Ensure positive definite matrices
    
    # Objective terms
    log_det_terms = sum(
        n_list[i] * (cp.log_det(Theta) - cp.trace(S[i] @ Theta)) 
        for i, Theta in enumerate(Thetas)
    )
    sparsity_terms = sum(cp.norm(Theta - cp.diag(cp.diag(Theta)), 1) for Theta in Thetas)
    
    # Smoothness term with user-specified ψ function
    smoothness_terms = sum(penalty_function(Thetas[i] - Thetas[i-1]) for i in range(1, T))
    
    # Objective function
    objective = cp.Maximize(log_det_terms - lambda_ * sparsity_terms - beta * smoothness_terms)
    
    # Problem setup
    problem = cp.Problem(objective, constraints)
    
    # Solve the problem
    problem.solve(solver=cp.SCS, verbose=True)
    
    # Extract results
    return [Theta.value for Theta in Thetas]

# Define ψ functions
def element_wise(X):
    """Element-wise L1 penalty for few edges changing."""
    return cp.norm(X, 1)

def group_lasso(X):
    """Group Lasso (L2 norm) penalty for global restructuring."""
    return cp.norm(X, 'fro')  # Frobenius norm as group lasso

def laplacian(X):
    """Laplacian penalty for smooth transitions."""
    return cp.sum_squares(X)

def infinity_norm(X):
    """L-infinity penalty for block-wise restructuring."""
    return cp.norm(X, "inf")

def perturbed_node(X):
    """Row-column overlap penalty for perturbed nodes."""
    
    V = cp.Variable(X.shape)  # Auxiliary variable
    constraints = [V + V.T == X]  # Ensure V + V^T equals X
    penalty = cp.sum(cp.norm(V, axis=0))  # Sum of column-wise L2 norms
    
    # Solve the sub-problem for V
    problem = cp.Problem(cp.Minimize(penalty), constraints)
    problem.solve(solver=cp.SCS)
    
    # Return the penalty value
    return problem.value
