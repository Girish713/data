import numpy as np
import pandas as pd

# Sample Data (features)
X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0]
])

## --- PCA Step 1: Standardization ---
# Calculate the mean and standard deviation for each feature
mean_vec = np.mean(X, axis=0)
std_vec = np.std(X, axis=0)

# Standardize the data (Z-score normalization)
X_std = (X - mean_vec) / std_vec
print("Standardized Data (X_std):\n", X_std)

## --- PCA Step 2: Covariance Matrix Calculation ---
# The covariance matrix is essential for finding the Principal Components (eigenvectors)
# Note: numpy.cov expects features as rows, so we transpose X_std
cov_mat = np.cov(X_std.T)
print("\nCovariance Matrix:\n", cov_mat)

# The next steps (Eigen Decomposition, Sorting, Projection) require advanced linear algebra functions,
# which are usually the part where a dedicated library is required for robustness.
