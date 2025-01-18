import numpy as np
import pandas as pd
import time
import penalty_functions as pf

class TVGL:

    np.set_printoptions(precision=3)

    def __init__(self, filename, lambd, beta, penalty_function="group_lasso"):
        """
        Initialize attributes and read data.

        Args:
            filename (str): Path to the CSV file.
            lambd (float): Regularization parameter.
            beta (float): Augmented Lagrangian parameter.
            penalty_function (str): Type of penalty function.
        """
        self.penalty_function = penalty_function
        self.dimension = None
        self.blocks = None
        self.emp_cov_mat = []
        self.read_data(filename)
        self.rho = self.get_rho()
        self.max_step = 0.1
        self.lambd = lambd
        self.beta = beta
        self.thetas = [np.ones((self.dimension, self.dimension))] * self.blocks
        self.z0s = [np.ones((self.dimension, self.dimension))] * self.blocks
        self.z1s = [np.ones((self.dimension, self.dimension))] * self.blocks
        self.z2s = [np.ones((self.dimension, self.dimension))] * self.blocks
        self.u0s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.u1s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.u2s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.obs=self.emp_cov_mat[0].shape[0]
        self.eta = float(self.obs) / float(3 * self.rho)
        self.e = 1e-5
        self.roundup = 1

    def read_data(self, filename):
        """
        Read data from a CSV file and divide it into blocks based on the last column.
        csv_file : raw,number,feature1,....feature10,block_number

        Args:
            filename (str): Path to the CSV file.
        """
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(filename)

        # Extract block labels (last column) and drop the first and last columns
        block_labels = data.iloc[:, -1].to_numpy()
        data = data.iloc[:, 1:-1].to_numpy()

        # Determine the number of features (columns) and unique blocks
        self.dimension = data.shape[1]
        unique_blocks = np.unique(block_labels)
        self.blocks = len(unique_blocks)

        # Initialize the empirical covariance matrices for each block
        self.emp_cov_mat = [0] * self.blocks

        # Group data by block and compute covariance matrices
        for block in unique_blocks:
            block_data = data[block_labels == block]
            tp = block_data.T
            self.emp_cov_mat[int(block)] = np.real(np.dot(tp, block_data) / block_data.shape[0])

    def get_rho(self):
        """Assign rho based on the number of observations in a block."""
        return float(self.emp_cov_mat[0].shape[0] + 0.1) / float(3)

    def run_algorithm(self, max_iter=10000):
        """
        Run the ADMM algorithm for TV Graphical Lasso.

        Args:
            max_iter (int): Maximum number of iterations.
        """
        self.iteration = 0
        stopping_criteria = False
        thetas_pre = []
        start_time = time.time()

        while self.iteration < max_iter and not stopping_criteria:
            if self.iteration % 500 == 0 or self.iteration == 1:
                print(f"\n*** Iteration {self.iteration} ***")
                print(f"Time passed: {time.time() - start_time:.3g}s")
                print(f"Rho: {self.rho}")

            self.theta_update()
            self.z_update()
            self.u_update()

            # Check stopping criteria
            if self.iteration > 0:
                fro_norm = sum(np.linalg.norm(self.thetas[i] - thetas_pre[i])for i in range(self.blocks))
                if fro_norm < self.e:
                    stopping_criteria = True

            thetas_pre = list(self.thetas)
            self.iteration += 1

        self.run_time = f"{time.time() - start_time:.3g}s"
        self.final_tuning(stopping_criteria, max_iter)

    def theta_update(self):
       for i in range(self.blocks):
            a = (self.z0s[i] + self.z1s[i] + self.z2s[i] -
                 self.u0s[i] - self.u1s[i] - self.u2s[i]) / 3
            at = a.transpose()
            m = self.eta * (a + at) / 2 - self.emp_cov_mat[i]
            d, q = np.linalg.eig(m)
            qt = q.transpose()
            sqrt_matrix = np.sqrt(d**2 + 4 / self.eta * np.ones(self.dimension))
            diagonal = np.diag(d) + np.diag(sqrt_matrix)
            self.thetas[i] = np.real(self.eta / 2 * np.dot(np.dot(q, diagonal), qt))

    def z_update(self):
        self.z0_update()
        self.z1_z2_update()
        
    def z0_update(self):
        self.z0s = [pf.soft_threshold_odd(
            self.thetas[i] + self.u0s[i], self.lambd, self.rho)
            for i in range(self.blocks)]
    
    def z1_z2_update(self):
        if self.penalty_function == "perturbed_node":
            for i in range(1, self.blocks):
                self.z1s[i - 1], self.z2s[i] = pf.perturbed_node(self.thetas[i - 1],
                                                                 self.thetas[i],
                                                                 self.u1s[i - 1],
                                                                 self.u2s[i],
                                                                 self.beta,
                                                                 self.rho)
        else:
            aa = [self.thetas[i] - self.thetas[i - 1] + self.u2s[i] - self.u1s[i - 1]
                  for i in range(1, self.blocks)]
            ee = [getattr(pf, self.penalty_function)(a, self.beta, self.rho)
                  for a in aa]
            for i in range(1, self.blocks):
                summ = self.thetas[i - 1] + self.thetas[i] + self.u1s[i - 1] + self.u2s[i]
                self.z1s[i - 1] = 0.5 * (summ - ee[i - 1])
                self.z2s[i] = 0.5 * (summ + ee[i - 1])

    def u_update(self):
        for i in range(self.blocks):
            self.u0s[i] = self.u0s[i] + self.thetas[i] - self.z0s[i]
        for i in range(1, self.blocks):
            self.u2s[i] = self.u2s[i] + self.thetas[i] - self.z2s[i]
            self.u1s[i - 1] = self.u1s[i - 1] + self.thetas[i - 1] - self.z1s[i - 1]

    
    def final_tuning(self, stopping_criteria, max_iter):
        """Perform final tuning for the converged thetas."""
        self.thetas = [np.round(theta, self.roundup) for theta in self.thetas]
        if stopping_criteria:
            print(f"\nIterations to complete: {self.iteration}")
        else:
            print(f"\nMax iterations ({max_iter}) reached")

    def temporal_deviations(self):
        """Compute temporal deviations between blocks."""
        self.deviations = np.zeros(self.blocks - 1)
        for i in range(self.blocks - 1):
            dif = self.thetas[i + 1] - self.thetas[i]
            np.fill_diagonal(dif, 0)
            self.deviations[i] = np.linalg.norm(dif)
        try:
            self.norm_deviations = self.deviations / max(self.deviations)
            self.dev_ratio = max(self.deviations) / np.mean(self.deviations)
        except ZeroDivisionError:
            self.norm_deviations = self.deviations
            self.dev_ratio = 0

    