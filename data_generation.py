import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import math

# Set a random seed for reproducibility
np.random.seed(42)

def generate_time_varying_gaussian_data(N, d, T):
    """
    Generates a DataFrame where observations follow a multivariate Gaussian distribution
    with a covariance matrix that changes over time.

    Parameters:
        - N: int, total number of observations
        - d: int, number of variables (features)
        - T: int, number of periods where the covariance matrix changes
    
    Returns:
        - data: pd.DataFrame, generated data
        - covariances: list of np.ndarray, the covariance matrices used
    """
    # Number of observations per period
    samples_per_period = N // T
    
    # Generate T positive definite covariance matrices
    covariances = []
    for _ in range(T):
        A = np.random.randn(d, d)
        cov_matrix = A @ A.T  # Ensure the matrix is positive definite
        covariances.append(cov_matrix)
    
    # Generate the data
    data = []
    for t in range(T):
        # Generate Gaussian data for period t
        mean = np.zeros(d)  # Zero mean
        covariance = covariances[t]
        samples = np.random.multivariate_normal(mean, covariance, samples_per_period)
        data.append(samples)
    
    # Stack the data and create a DataFrame
    data = np.vstack(data)
    columns = [f"Feature_{i+1}" for i in range(d)]
    df = pd.DataFrame(data, columns=columns)
    
    return df, covariances

def plot_heatmaps(covariances):
    """
    Visualizes the evolution of covariance matrices using heatmaps.

    Parameters:
        - covariances: list of np.ndarray, the covariance matrices.
    """
    T = len(covariances)
    plt.figure(figsize=(15, 5))
    
    for t, sigma in enumerate(covariances):
        plt.subplot(1, T, t + 1)
        sns.heatmap(sigma, annot=False, cmap="coolwarm", cbar=False, square=True)
        plt.title(f"{t + 1}")
        plt.tight_layout()

def generate_time_varying_gaussian_data_with_perturbation(N, d, T, t_shift, delta):
    """
    Generates a DataFrame where observations follow a multivariate Gaussian distribution
    with a covariance matrix that changes over time, including a perturbation at a specific moment.

    Parameters:
        - N: int, total number of observations
        - d: int, number of variables (features)
        - T: int, number of periods where the covariance matrix changes
        - t_shift: int, period where the perturbation occurs
        - delta: float, intensity of the correlation change
    
    Returns:
        - data: pd.DataFrame, generated data
        - covariances: list of np.ndarray, the covariance matrices used
    """
    # Number of observations per period
    samples_per_period = N // T
    
    # Generate the initial covariance matrix
    A = np.random.randn(d, d)
    Sigma_0 = A @ A.T  # Ensure it is positive definite
    
    # Normalize to obtain correlations
    std_devs = np.sqrt(np.diag(Sigma_0))
    Sigma_0 = Sigma_0 / np.outer(std_devs, std_devs)
    
    # Copy Sigma_0 for all periods
    covariances = [Sigma_0.copy() for _ in range(T)]
    
    # Modify a variable at the t_shift period
    if t_shift < T:
        Sigma_shift = covariances[t_shift].copy()
        variable_to_modify = np.random.randint(0, d)
        
        # Change correlations of this variable with others
        for i in range(d):
            if i != variable_to_modify:
                Sigma_shift[variable_to_modify, i] += delta
                Sigma_shift[i, variable_to_modify] += delta
        
        # Re-enforce covariance matrix properties
        std_devs = np.sqrt(np.diag(Sigma_shift))
        Sigma_shift = Sigma_shift / np.outer(std_devs, std_devs)
        covariances[t_shift] = Sigma_shift
    
    # Generate the data
    data = []
    for t in range(T):
        mean = np.zeros(d)
        covariance = covariances[t]
        samples = np.random.multivariate_normal(mean, covariance, samples_per_period)
        period_data = pd.DataFrame(samples, columns=[f"Feature_{i+1}" for i in range(d)])
        period_data['Period'] = t
        data.append(period_data)
    
    # Concatenate the data into a final DataFrame
    return pd.concat(data, ignore_index=True), covariances

def plot_graph(covariances, threshold=0.1):
    """
    Visualizes covariance matrices as weighted graphs in subplots.

    Parameters:
        - covariances: list of np.ndarray, the covariance matrices.
        - threshold: float, minimum value to include an edge in the graph.
    """
    T = len(covariances)  # Number of periods
    cols = 3  # Number of columns for subplots
    rows = math.ceil(T / cols)  # Calculate rows based on total periods and columns

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten axes for easier iteration if rows > 1

    for t, sigma in enumerate(covariances):
        ax = axes[t]  # Get the corresponding subplot
        G = nx.Graph()
        
        # Add nodes and edges
        d = sigma.shape[0]
        for i in range(d):
            G.add_node(i)
            for j in range(i + 1, d):
                if abs(sigma[i, j]) > threshold:
                    G.add_edge(i, j, weight=sigma[i, j])
        
        # Draw the graph
        pos = nx.circular_layout(G)
        edges = G.edges(data=True)
        nx.draw(
            G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, ax=ax
        )
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels={(u, v): f"{d['weight']:.2f}" for u, v, d in edges}, ax=ax
        )
        ax.set_title(f"Graph of Covariance (Period {t + 1})")
    
    # Turn off unused axes
    for i in range(T, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()
