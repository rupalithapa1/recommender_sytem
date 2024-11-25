
import pandas as pd
from data_preparation import prepare_data, split_data
from matrix_factorization import apply_nmf, apply_svd
from recommendation import recommend_items
from evaluation_metrics import calculate_rmse, precision_at_k

# Load data
user_item_matrix_path = "user_item_matrix.csv"
user_item_matrix = prepare_data(user_item_matrix_path)

# Split data
train, test = split_data(user_item_matrix)

# Apply NMF
_, _, reconstructed_matrix = apply_nmf(train)

# Example: Recommendations for a specific user
user_id = "U00332"
recommendations = recommend_items(user_id, train, reconstructed_matrix)
print("Recommendations:")
print(recommendations)

# Evaluate
rmse = calculate_rmse(test.values, reconstructed_matrix)
print(f"RMSE: {rmse}")
