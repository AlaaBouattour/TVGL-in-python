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
    return cp.norm(X, 1),[]

def group_lasso(X):
    """Group Lasso (L2 norm) penalty for global restructuring."""
    return cp.norm(X, 'fro'),[]  # Frobenius norm as group lasso

def laplacian(X):
    """Laplacian penalty for smooth transitions."""
    return cp.sum_squares(X),[]

def infinity_norm(X):
    """L-infinity penalty for block-wise restructuring."""
    return cp.norm(X, "inf"),[]

def perturbed_node(X):
    """Row-column overlap penalty for perturbed nodes."""
    p, _ = X.shape
    V = cp.Variable((p, p))  
    constraints = [V + V.T == X]
    penalty_expr = cp.sum(cp.norm(V, axis=0))  # sum_j || V[:,j] ||_2
    return penalty_expr, constraints



def solve_time_varying_graphical_lasso(
    S, n_list, lambda_, beta, penalty_function, p, T, 
    solver=cp.SCS, verbose=True
):
    """
    Solves the time-varying graphical lasso problem with a user-specified penalty function
    that returns (expr, constraints).

    Parameters
    ----------
    S : list of np.ndarray
        List of empirical covariance matrices (length T).
    n_list : list of int
        List of sample sizes for each time step (length T).
    lambda_ : float
        Regularization parameter for sparsity.
    beta : float
        Regularization parameter for smoothness.
    penalty_function : callable
        A function that takes a CVXPY expression X and returns (expr, constraints).
        Examples: element_wise, group_lasso, laplacian, etc.
    p : int
        Size of the covariance matrix (number of variables).
    T : int
        Number of time steps.
    solver : str, optional
        Which solver to use. Default: cp.SCS
    verbose : bool, optional
        If True, prints solver output.

    Returns
    -------
    Thetas_value : list of np.ndarray
        List of precision matrices (Theta) for each time step.
    problem_value : float
        Final value of the objective function.
    """
    # 1) Create variables: Theta_t for t=1..T
    #    We can specify PSD=True to ensure each is positive semidefinite
    Thetas = [cp.Variable((p, p), PSD=True) for _ in range(T)]

    # 2) Basic constraints: Theta_t >> 0
    #    (Using PSD=True already imposes that, but let's be explicit.)
    constraints = []
    for Theta in Thetas:
        constraints.append(Theta >> 0)

    # 3) Build the log-likelihood sum:
    #    sum_{i=1..T} [ n_i * ( log_det(Theta_i) - trace(S_i @ Theta_i ) ) ]
    log_likelihood = 0
    for i in range(T):
        log_likelihood += n_list[i] * (
            cp.log_det(Thetas[i]) - cp.trace(S[i] @ Thetas[i])
        )

    # 4) Sparsity penalty: sum_{i=1..T} || offdiag(Theta_i)||_1
    #    offdiag(Theta_i) = Theta_i - diag(diag(Theta_i))
    sparsity_expr = 0
    for i in range(T):
        diag_part = cp.diag(cp.diag(Thetas[i]))
        offdiag = Thetas[i] - diag_part
        sparsity_expr += cp.norm(offdiag, 1)

    # 5) Smoothness term: sum_{i=2..T} psi(Theta_i - Theta_{i-1})
    #    penalty_function returns (expr, extra_cons)
    smooth_expr = 0
    extra_cons = []
    for i in range(1, T):
        expr_i, cons_i = penalty_function(Thetas[i] - Thetas[i-1])
        smooth_expr += expr_i
        extra_cons.extend(cons_i)

    # Add those constraints to the main list
    constraints += extra_cons

    # 6) Final objective is to maximize the log-likelihood minus penalties,
    #    or equivalently: Maximize( log_likelihood ) - lambda * sparsity_expr - beta * smooth_expr
    #    In CVXPY we define:
    objective = cp.Maximize(
        log_likelihood 
        - lambda_ * sparsity_expr
        - beta * smooth_expr
    )

    # 7) Solve
    problem = cp.Problem(objective, constraints)
    result = problem.solve(solver=solver, verbose=verbose)

    # 8) Extract solutions
    Thetas_value = [Theta.value for Theta in Thetas]
    return Thetas_value
