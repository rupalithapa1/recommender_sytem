import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib

class FeatureEngineering:
    def __init__(self, train_data_path, root_dir):
        """
        Initialize the FeatureEngineering class.
        
        Args:
            train_data_path (str): Path to the training data file.
            root_dir (str): Directory where transformed data and pipelines will be saved.
        """
        self.train_data_path = train_data_path
        self.root_dir = root_dir

    def load_and_preprocess_data(self):
        """
        Load and preprocess the dataset from a CSV file.

        Returns:
            pd.DataFrame: Preprocessed dataset.
        """
        try:
            # Load the dataset
            df = pd.read_csv(self.train_data_path)
            
            # Drop irrelevant columns
            if "Item_Identifier" in df.columns:
                df.drop("Item_Identifier", axis=1, inplace=True)
            
            # Handle missing values
            if df.isnull().sum().sum() > 0:
                df.fillna(df.mean(), inplace=True)
            
            # Add derived features
            if "Outlet_Establishment_Year" in df.columns:
                df["Years_Operational"] = 2024 - df["Outlet_Establishment_Year"]
            
            return df
        except Exception as e:
            raise Exception(f"Error in load_and_preprocess_data: {e}")

    def transform_features(self):
        """
        Transform features using preprocessing pipelines.

        Returns:
            np.ndarray: Transformed feature matrix.
            np.ndarray or None: Target variable (if present).
        """
        try:
            df = self.load_and_preprocess_data()

            # Define feature types
            numerical_features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
            categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

            # Separate target variable if present
            if "target" in df.columns:
                y = df["target"]
                df.drop(columns=["target"], inplace=True)
            else:
                y = None

            X = df

            # Define preprocessing pipelines
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine transformations
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )

            # Full pipeline with PCA
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('pca', PCA(n_components=None))  # Set components if needed
            ])

            # Fit and transform the data
            X_transformed = pipeline.fit_transform(X)

            # Save the pipeline
            pipeline_path = os.path.join(self.root_dir, "pipeline.joblib")
            joblib.dump(pipeline, pipeline_path)

            return X_transformed, y
        except Exception as e:
            raise Exception(f"Error in transform_features: {e}")

    def save_transformed_data(self, X_transformed, y):
        """
        Save transformed data to .npz files.

        Args:
            X_transformed (np.ndarray): Transformed feature matrix.
            y (np.ndarray or None): Target variable (if present).
        """
        try:
            transformed_data_path = self.root_dir
            np.savez(os.path.join(transformed_data_path, 'Transformed_Data.npz'), 
                     X_transformed=X_transformed, y=y)
        except Exception as e:
            raise Exception(f"Error in save_transformed_data: {e}")

# Example usage
if __name__ == "__main__":
    train_data_path = "path/to/your/train_data.csv"  # Replace with actual path
    root_dir = "path/to/save/pipeline_and_data"     # Replace with actual path

    feature_engineering = FeatureEngineering(train_data_path, root_dir)
    X_transformed, y = feature_engineering.transform_features()
    feature_engineering.save_transformed_data(X_transformed, y)

