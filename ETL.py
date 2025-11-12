import pandas as pd

# --- Extraction (E) ---
# Assuming 'raw_data.csv' exists (or use sample creation)
# raw_data = pd.read_csv('raw_data.csv')

# Use in-memory sample for demonstration
raw_data = pd.DataFrame({
    'CustomerID': [101, 102, 103, 104, 105],
    'Age': [25, 45, 19, 30, np.nan],
    'Income': ['50k', '75k', '30k', 'NA', '120k'],
    'Product_ID': ['A1', 'B2', 'A1', 'C3', 'B2']
})

# --- Transformation (T) ---
# 1. Handle missing values (Impute missing Age with the mean)
mean_age = raw_data['Age'].mean()
raw_data['Age'].fillna(mean_age, inplace=True)

# 2. Clean 'Income' column (Convert to numeric and handle 'NA')
cleaned_income = (raw_data['Income']
                  .str.replace('k', '')
                  .replace('NA', np.nan)
                  .astype(float) * 1000)
raw_data['Income'] = cleaned_income.fillna(cleaned_income.median())

# 3. Create a new feature (Categorize Age)
raw_data['Age_Group'] = pd.cut(raw_data['Age'], bins=[0, 25, 50, 100], labels=['Young', 'Middle', 'Senior'])

transformed_data = raw_data
print("--- Transformed Data ---\n", transformed_data)

# --- Loading (L) ---
# Save the clean, transformed data to a new CSV file
# transformed_data.to_csv('cleaned_data.csv', index=False)
# print("\nData loaded to 'cleaned_data.csv'")
