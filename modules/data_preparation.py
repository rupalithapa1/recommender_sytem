import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(user_item_matrix_path):
    # Load user-item matrix
    user_item_matrix = pd.read_csv(user_item_matrix_path, index_col=0)
    return user_item_matrix

def split_data(user_item_matrix, test_size=0.2, random_state=42):
    # Split into training and test sets
    train, test = train_test_split(user_item_matrix, test_size=test_size, random_state=random_state)
    # Ensure they have the same structure as the original
    train = pd.DataFrame(train, index=user_item_matrix.index, columns=user_item_matrix.columns)
    test = pd.DataFrame(test, index=user_item_matrix.index, columns=user_item_matrix.columns)
    return train, test