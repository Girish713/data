import numpy as np
import pandas as pd

# Sample Data
data = pd.DataFrame({
    'Feature1': [1, 2, 3, 6, 7, 8],
    'Feature2': [10, 12, 11, 20, 21, 23],
    'Class': [0, 0, 0, 1, 1, 1]
})

X = data[['Feature1', 'Feature2']].values
y = data['Class'].values

# --- LDA Step 1: Calculate Class Means ---
class_means = []
for cl in np.unique(y):
    mean_vector = np.mean(X[y == cl], axis=0)
    class_means.append(mean_vector)
    print(f"Mean Vector for Class {cl}: {mean_vector}")

## --- LDA Step 2: Calculate Within-Class Scatter Matrix (Sw) ---
# Sw is the sum of scatter matrices for each class (Si)
d = X.shape[1]  # number of features
Sw = np.zeros((d, d))

for cl, mean_vec in enumerate(class_means):
    # Calculate Si (Scatter Matrix for class)
    X_cl = X[y == cl]
    S_i = np.zeros((d, d))
    for row in X_cl:
        row = row.reshape(d, 1)  # make it a column vector
        mean_vec = mean_vec.reshape(d, 1)
        S_i += (row - mean_vec).dot((row - mean_vec).T)
    Sw += S_i

print("\nWithin-Class Scatter Matrix (Sw):\n", Sw)

# The next steps (Between-Class Scatter Matrix, Generalized Eigenvalue problem) are complex.
