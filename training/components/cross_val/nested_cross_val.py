from training.exception import NestedCrossValError,handle_exception
from training.custom_logging import info_logger, error_logger
from training.entity.config_entity import NestedCrossValConfig
from training.configuration_manager.configuration import ConfigurationManager

import os
import sys
import cv2
from joblib import dump
from joblib import load
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, loguniform, randint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report



class NestedCrossVal:
    def __init__(self, config: NestedCrossValConfig) -> None:
        self.config = config

    @staticmethod
    def is_json_serializable(value):
      """
      Check if a value is JSON serializable.
      """
      try:
          json.dumps(value)
          return True
      except (TypeError, OverflowError):
          return False

    def get_data_labels_groups(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads the extracted features, labels and groups .
            Returns the extracted features, labels and groups : X,y and groups
            """
        try:
            # Paths to the .npz files
            output_dir = self.config.extracted_features
            data_path = os.path.join(output_dir, 'X.npz')
            labels_path = os.path.join(output_dir, 'y.npz')
            groups_path = os.path.join(output_dir, 'groups.npz')

            # Loading the .npz files
            data = np.load(data_path)
            labels = np.load(labels_path)
            groups = np.load(groups_path)

            #  Access the arrays stored inside the .npz files
            X = data['data']
            y = labels['labels']
            groups = groups['groups']

            return X, y, groups
        except Exception as e:
            handle_exception(e, NestedCrossValError)
    

    def train_test_split(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,np.ndarray]:
        """
        - Splits the data into train and test sets based on the specified groups and the specified test_size.
        - Splitting the original dataset into train(70%) and test(20%) using GroupShuffleSplit to ensure that 
          the same group of images are either in the train or the test dataset only.
        """
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]

            return X_train, X_test, y_train, y_test,groups_train
        except Exception as e:
            handle_exception(e, NestedCrossValError)
        

    def save_train_test_data_for_final_train(self, X_train, X_test, y_train, y_test, groups_train):
        try:
            # Save X_train, X_test, y_train, y_test to be used by final_train
            # Path to the 'Extracted Features' directory

            train_data_path = self.config.train_data_path
            test_data_path = self.config.test_data_path

            # Save X_train,y_train and groups_train to train.npz
            # Save X_test,y_test to test.npz
            np.savez(os.path.join(train_data_path, 'Train.npz'), X_train=X_train, y_train=y_train, groups_train=groups_train)
            np.savez(os.path.join(test_data_path, 'Test.npz'),  X_test=X_test, y_test=y_test)

            info_logger.info(f"Train.npz and Test.npz saved at {train_data_path} and {test_data_path}")
        except Exception as e:
            handle_exception(e, NestedCrossValError)

    def initialize_outer_loop(self):
        try:
            outer_cv = GroupKFold(n_splits=2)
            return outer_cv
        except Exception as e:
            handle_exception(e, NestedCrossValError)
    
    def start_inner_loop(self,count,X_outer_train, y_outer_train, groups_outer_train, X_outer_val, y_outer_val):
        
        try:
            if not os.path.exists(self.config.random_search_models_rf):
                os.makedirs(self.config.random_search_models_rf)
            # Defining inner cross-validation strategy
            inner_cv = GroupKFold(n_splits=3)

            # Creating pipelines and parameters grids for RF models
            pipeline_rf = self.create_pipelines()
            param_grid_rf = self.create_param_grids()

            # Performing RandomSearchCV for RF
            random_search_rf = self.randomized_search_rf(count,pipeline_rf,param_grid_rf,inner_cv,X_outer_train,y_outer_train,groups_outer_train)
            best_model_rf = self.get_best_model_rf(random_search_rf)
            self.evaluate_best_model_rf(count=count, best_model_rf=best_model_rf, X_outer_val=X_outer_val, y_outer_val=y_outer_val)
            dump(best_model_rf,f"{self.config.random_search_models_rf}/rf_{count}.joblib")

            with open(self.config.STATUS_FILE, "a") as f:
                f.write(f"Model {str(count)} saved\n")
            del best_model_rf

        except Exception as e:
            handle_exception(e, NestedCrossValError)


    def create_pipelines(self):
        # Pipeline for Random Forest Classifier
        pipeline_rf = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",PCA(n_components=None)),
            ("classifier", RandomForestClassifier())
        ])

        # Similary Create Pipeline for SVM
        # ...

        return pipeline_rf
    

    # Creating Parameter Grids
    def create_param_grids(self):

        # Used Wolfram to decide mean and std_dev https://www.wolframalpha.com/input?i=X%7EN%28450%2C40%5E2%29+P%28300%3C%3DX%3C%3D500%29
        # Define parameters for the normal distribution
        mean = 550      # Center of the distribution
        std_dev =40    # Standard deviation (controls spread)

        # Generate a normal distribution of float values
        values = np.random.normal(loc=mean, scale=std_dev, size=1000)

        # Clip the values to ensure they lie within the range [300, 500]
        values = np.clip(values, 500, 600)

        # Convert the float values to integers
        n_estimators = values.astype(int)

        # Randomly sample from the integer values (e.g., sample 100 values)
        #n_estimators = np.random.choice(integer_values, size=100, replace=False)

        # Parameter grid for Random Forest Classifier
        param_grid_rf = {
            'classifier__n_estimators':n_estimators,
            'classifier__max_depth':randint(30,40),
            'classifier__min_samples_split': randint(100,200)
        }

        return param_grid_rf
    

        # Similary create parameter grid for SVM


    def randomized_search_rf(self,count,pipeline_rf, param_grid_rf, inner_cv, X_outer_train, y_outer_train, groups_outer_train):
        """
        Perform RandomizedSearch with Cross-validation for Hyperparameter Tuning
        """

        cache_dir = self.config.model_cache_rf
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Use joblib to cache RandomizedSearchCV
        random_search_rf = RandomizedSearchCV(
            estimator=pipeline_rf,
            param_distributions=param_grid_rf,
            cv = inner_cv,
            scoring="f1_macro",
            n_iter=3,
            n_jobs=-1, # Uses all processors
            verbose=4 
        )

        cached_search = os.path.join(cache_dir, f'random_search_rf_{count}.joblib')
        if os.path.exists(cached_search):
            with open(self.config.STATUS_FILE, "a") as f:
                f.write("Loading cached model\n")
            random_search_rf = load(cached_search)
        else:
            random_search_rf.fit(X_outer_train, y_outer_train, groups=groups_outer_train)
            with open(self.config.STATUS_FILE, "a") as f:
                f.write(f"Fitting Model {str(count)}\n")
            dump(random_search_rf, cached_search)

        return random_search_rf


    def get_best_model_rf(self,random_search_rf):
        best_model_rf = random_search_rf.best_estimator_
        best_params_rf = random_search_rf.best_params_
        best_scores_rf = random_search_rf.best_score_

        with open(self.config.STATUS_FILE, "a") as f:
            f.write(f"Best params for RF Model: {str(best_params_rf)}\n")
            f.write(f"Best scoring(F1) for RF Model: {str(best_scores_rf)}\n")

        return best_model_rf


    def evaluate_best_model_rf(self,count,best_model_rf,X_outer_val,y_outer_val):
        """
        Evaluate Best Model from RandomizedSearch on Outer Validation Fold
        """
        y_outer_val_pred_rf = best_model_rf.predict(X_outer_val)

        report_dict = classification_report(y_outer_val, y_outer_val_pred_rf, output_dict=True)

        # Saving the metrics as a json file
        metric_file_path = os.path.join(self.config.metric_file_name_rf, f'metrics_rf_{count}.json')
        with open(metric_file_path, 'w') as f:
            json.dump(report_dict, f, indent=4)

        
        # Saving the best model params as a json file
        best_model_rf_params_path = os.path.join(self.config.best_model_params_rf, f'best_params_rf_{count}.json')
        best_model_rf_params = best_model_rf.get_params()
        serializable_params = {k: v for k, v in best_model_rf_params.items() if self.is_json_serializable(v)}
        
        with open(best_model_rf_params_path, 'w') as f:
            json.dump(serializable_params, f, indent=4)