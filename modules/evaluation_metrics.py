from sklearn.metrics import mean_squared_error

def calculate_rmse(original_matrix, reconstructed_matrix):
    return np.sqrt(mean_squared_error(original_matrix, reconstructed_matrix))

def precision_at_k(recommendations, test_data, k=7):
    relevant_items = test_data[test_data > 0].index.tolist()
    recommended_items = recommendations['ItemID'][:k].tolist()
    precision = len(set(recommended_items).intersection(set(relevant_items))) / k
    return precision
