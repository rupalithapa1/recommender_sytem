import pandas as pd

def recommend_items(user_id, user_item_matrix, reconstructed_matrix, n_recommendations=7):
    user_index = user_item_matrix.index.get_loc(user_id)
    user_ratings = reconstructed_matrix[user_index]
    rated_items = user_item_matrix.columns[user_item_matrix.iloc[user_index] > 0].tolist()

    # Recommend for unrated items
    unrated_indices = [i for i, item in enumerate(user_item_matrix.columns) if item not in rated_items]
    recommendations = [(user_item_matrix.columns[i], user_ratings[i]) for i in unrated_indices]
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:n_recommendations]
    return pd.DataFrame(recommendations, columns=['ItemID', 'PredictedRating'])
