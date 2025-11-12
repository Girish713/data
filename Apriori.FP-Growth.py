import pandas as pd

# Sample Retail Transaction Data
raw_transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Milk', 'Sugar'],
    ['Milk', 'Bread'],
    ['Bread', 'Butter', 'Eggs'],
    ['Sugar', 'Eggs']
]

# Convert the list of transactions into a suitable format for the algorithm
# We need a list of lists or a one-hot encoded DataFrame.

## --- Option 1: List of Lists (Input for custom algorithm) ---
transactions_list = raw_transactions
print("Input Transaction List:\n", transactions_list)

## --- Option 2: One-Hot Encoded DataFrame (For support calculation) ---
# Get all unique items
all_items = sorted(list(set(item for sublist in raw_transactions for item in sublist)))

# Create an empty DataFrame
data = {}
for item in all_items:
    data[item] = []

# Populate the DataFrame
for transaction in raw_transactions:
    for item in all_items:
        data[item].append(1 if item in transaction else 0)

ohe_df = pd.DataFrame(data)
print("\nOne-Hot Encoded Transaction Data:\n", ohe_df)

# --- Core Logic for Support Calculation (Apriori/FP-Growth) ---
# Calculate the support for a single item (e.g., 'Milk')
support_milk = ohe_df['Milk'].mean()
print(f"\nSupport (Milk): {support_milk}")

# Calculate the support for an itemset (e.g., {'Milk', 'Bread'})
# This is the non-zero count of the intersection / Total Transactions
support_milk_bread = (ohe_df['Milk'] & ohe_df['Bread']).mean()
print(f"Support (Milk, Bread): {support_milk_bread}")

# The complete Apriori/FP-Growth algorithm requires iterative candidate generation and pruning/tree building.
