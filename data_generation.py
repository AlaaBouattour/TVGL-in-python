import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def generate_time_varying_gaussian_data(N, d, T):
    """
    Génère un DataFrame où les observations suivent une loi gaussienne multivariée
    avec une matrice de covariance variant au cours du temps.

    Parameters:
        - N: int, nombre total d'observations
        - d: int, nombre de variables (features)
        - T: int, nombre de périodes où la matrice de covariance change
    
    Returns:
        - data: pd.DataFrame, données générées
        - covariances: list of np.ndarray, les matrices de covariance utilisées
    """
    # Nombre d'observations par période
    samples_per_period = N // T
    
    # Générer T matrices de covariance définies positives
    covariances = []
    for _ in range(T):
        A = np.random.randn(d, d)
        cov_matrix = A @ A.T  # Matrice définie positive
        covariances.append(cov_matrix)
    
    # Générer les données
    data = []
    for t in range(T):
        # Générer des données gaussiennes pour la période t
        mean = np.zeros(d)  # Moyenne nulle
        covariance = covariances[t]
        samples = np.random.multivariate_normal(mean, covariance, samples_per_period)
        data.append(samples)
    
    # Empiler les données et créer un DataFrame
    data = np.vstack(data)
    columns = [f"Feature_{i+1}" for i in range(d)]
    df = pd.DataFrame(data, columns=columns)
    
    return df, covariances

def plot_covariance_heatmaps(covariances):
    """
    Visualise l'évolution des matrices de covariance en utilisant des heatmaps.

    Parameters:
        - covariances: list of np.ndarray, les matrices de covariance.
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
    Génère un DataFrame où les observations suivent une loi gaussienne multivariée
    avec une matrice de covariance variant au cours du temps, incluant une perturbation
    à un moment donné.

    Parameters:
        - N: int, nombre total d'observations
        - d: int, nombre de variables (features)
        - T: int, nombre de périodes où la matrice de covariance change
        - t_shift: int, période où la perturbation a lieu
        - delta: float, intensité du changement de corrélation
    
    Returns:
        - data: pd.DataFrame, données générées
        - covariances: list of np.ndarray, les matrices de covariance utilisées
    """
    # Nombre d'observations par période
    samples_per_period = N // T
    
    # Générer la matrice de covariance initiale
    A = np.random.randn(d, d)
    Sigma_0 = A @ A.T  # Assurer qu'elle est définie positive
    
    # Normaliser pour obtenir des corrélations
    std_devs = np.sqrt(np.diag(Sigma_0))
    Sigma_0 = Sigma_0 / np.outer(std_devs, std_devs)
    
    # Copier Sigma_0 pour toutes les périodes
    covariances = [Sigma_0.copy() for _ in range(T)]
    
    # Modifier une variable à la période t_shift
    if t_shift < T:
        Sigma_shift = covariances[t_shift].copy()
        variable_to_modify = np.random.randint(0, d)
        
        # Changer les corrélations de cette variable avec les autres
        for i in range(d):
            if i != variable_to_modify:
                Sigma_shift[variable_to_modify, i] += delta
                Sigma_shift[i, variable_to_modify] += delta
        
        # Ré-imposer les propriétés de covariance
        std_devs = np.sqrt(np.diag(Sigma_shift))
        Sigma_shift = Sigma_shift / np.outer(std_devs, std_devs)
        covariances[t_shift] = Sigma_shift
    
    # Générer les données
    data = []
    for t in range(T):
        mean = np.zeros(d)
        covariance = covariances[t]
        samples = np.random.multivariate_normal(mean, covariance, samples_per_period)
        period_data = pd.DataFrame(samples, columns=[f"Feature_{i+1}" for i in range(d)])
        period_data['Period'] = t
        data.append(period_data)
    
    # Empiler les données dans un DataFrame final
    return pd.concat(data, ignore_index=True), covariances
