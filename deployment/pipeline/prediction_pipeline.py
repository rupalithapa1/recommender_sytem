import pandas as pd
import pickle

class PredictionPipeline:
    def __init__(self):
        # Load the reconstructed matrix (model output)
        with open('reconstructed_matrix.pkl', 'rb') as file:
            self.reconstructed_matrix = pickle.load(file)

        # Load the user-item matrix for reference
        self.user_item_matrix = pd.read_csv('user_item_matrix.csv', index_col=0)

    def recommend_items(self, user_id, n_recommendations=5):
        """Recommend top-N items for a given user."""
        # Find the user's row in the user-item matrix
        if user_id not in self.user_item_matrix.index:
            return ["User ID not found!"]

        user_index = self.user_item_matrix.index.get_loc(user_id)
        user_ratings = self.reconstructed_matrix[user_index]

        # Find items the user hasn't rated yet
        rated_items = self.user_item_matrix.columns[self.user_item_matrix.iloc[user_index] > 0].tolist()
        unrated_indices = [i for i, item in enumerate(self.user_item_matrix.columns) if item not in rated_items]

        # Predict ratings for unrated items and sort by predicted rating
        recommendations = [(self.user_item_matrix.columns[i], user_ratings[i]) for i in unrated_indices]
        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:n_recommendations]

        # Return the top-N recommendations as item names
        return [item[0] for item in recommendations]

    def predict(self, user_id, ratings=None):
        """Predict recommendations based on user_id."""
        return self.recommend_items(user_id)
