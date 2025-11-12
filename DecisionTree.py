import numpy as np

# Sample target variable array (0 = No, 1 = Yes)
y_data = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])

## --- Step 1: Calculate Probability for each class ---
total_samples = len(y_data)
unique_classes = np.unique(y_data)
probabilities = {}

for c in unique_classes:
    count_c = np.sum(y_data == c)
    probabilities[c] = count_c / total_samples

print(f"Class Probabilities: {probabilities}")

## --- Step 2: Calculate Gini Impurity (Gini(D)) ---
# Gini(D) = 1 - sum(p_k^2)
gini_impurity = 1 - sum(p**2 for p in probabilities.values())

print(f"\nGini Impurity (Gini(D)): {gini_impurity}")

## --- Step 3: Evaluate Model Accuracy (A standard metric) ---
# Example: True labels (y_true) and predicted labels (y_pred)
y_true = np.array([0, 1, 0, 1, 0])
y_pred = np.array([0, 1, 1, 1, 0])

# Calculate Accuracy: (True Positives + True Negatives) / Total
accuracy = np.mean(y_true == y_pred)
print(f"\nModel Accuracy: {accuracy}")
