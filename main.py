# main.py
import pandas as pd
from preprocessing import preprocess
from tvgl_solver import solve_tvgl
from config import WINDOW_SIZE, LAMBDA, BETA, RHO, MAX_ITER, TOL

def main():
    # Load dataset (replace with your actual dataset)
    data = pd.read_csv("time_series.csv")
    
    # Preprocess data
    covariances = preprocess(data, WINDOW_SIZE)
    
    # Solve TVGL
    precision_matrices = solve_tvgl(covariances, LAMBDA, BETA, RHO, MAX_ITER, TOL)
    
    # Output results
    for t, theta in enumerate(precision_matrices):
        print(f"Precision matrix at time {t}:\n", theta)

if __name__ == "__main__":
    main()
