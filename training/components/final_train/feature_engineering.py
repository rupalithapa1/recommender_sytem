import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
from training.exception import FeatureEngineeringError,handle_exception
from training.custom_logging import info_logger, error_logger
from training.entity.config_entity import FeatureEngineeringConfig
from training.configuration_manager.configuration import ConfigurationManager
import os

def load_data(file_path):
    """Load data from a given file path."""
    return pd.read_excel(file_path)

def drop_columns(df, columns_to_drop):
    """Drop specified columns from the DataFrame."""
    return df.drop(columns=columns_to_drop, errors='ignore')

def assign_random_items(item_list, n=3):
    """Assign a random list of items from a given list."""
    return list(np.random.choice(item_list, size=n, replace=False))

def feature_engineering(df):
    """Apply feature engineering transformations to the DataFrame."""
    # Extract unique ItemIDs for random sampling
    unique_item_ids = df['ItemID'].unique()
    
    # Add new columns with random ItemIDs
    df['CartActivity'] = df['UserID'].apply(lambda _: assign_random_items(unique_item_ids))
    df['WishlistActivity'] = df['UserID'].apply(lambda _: assign_random_items(unique_item_ids))
    df['AbandonedCartData'] = df['UserID'].apply(lambda _: assign_random_items(unique_item_ids))
    df['BrowsingHistory'] = df['UserID'].apply(lambda _: assign_random_items(unique_item_ids))
    
    return df

def save_data(df, output_file):
    """Save the DataFrame to a specified file."""
    df.to_csv(output_file, index=False)
