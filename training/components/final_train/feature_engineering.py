import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from training.exception import FeatureEngineeringError,handle_exception
from training.custom_logging import info_logger, error_logger
from training.entity.config_entity import FeatureEngineeringConfig
from training.configuration_manager.configuration import ConfigurationManager


class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def transform_features(self):
            
        try:
            train_data_path = os.path.join(self.config.train_data_path,"Train.npz")
            test_data_path = os.path.join(self.config.test_data_path,"Test.npz")
            # Loading the .npz files
            train_data = np.load(train_data_path)
            test_data = np.load(test_data_path)

            X_train, y_train, groups_train = train_data["X_train"], train_data["y_train"], train_data["groups_train"]
            X_test, y_test = test_data["X_test"], test_data["y_test"]

            info_logger.info(f"Data Split by NestedCrossVal loaded from {train_data_path} and {test_data_path}")
            info_logger.info(f"{37} Dtype of X_train: {X_train.dtype} and X_test is {X_test.dtype}")
            info_logger.info(f"{38} Dtype of y_train: {y_train.dtype} and y_test is {y_test.dtype}")

            return self.check_transformed_already_exists(X_train,X_test, y_train, y_test, groups_train)
        except Exception as e:
            handle_exception(e, FeatureEngineeringError)
    

    def proceed_with_feature_engineering(self, X_train, X_test, y_train, y_test, groups_train):
        info_logger.info("Features are now BEING transformed")
        transform_pipeline = Pipeline([
                  ("scaler", StandardScaler()),
                  ("pca",PCA(n_components=None))
              ])
        transform_pipeline.fit(X_train)

        # Save the pipeline
        pipeline_path = os.path.join(self.config.root_dir,"pipeline.joblib")
        joblib.dump(transform_pipeline, pipeline_path)
  

        X_train = transform_pipeline.transform(X_train)
        X_test = transform_pipeline.transform(X_test)

        info_logger.info("Features NOW transformed")
        
        with open(self.config.STATUS_FILE,"w") as f:
            f.write(f"Feature engineering completed {True}")
            
        return X_train, X_test, y_train, y_test, groups_train
    
    def check_transformed_already_exists(self, X_train, X_test, y_train, y_test, groups_train):
      if os.path.exists(os.path.join(self.config.root_dir,"Train.npz")):
          if os.path.exists(os.path.join(self.config.root_dir,"Test.npz")):
              
              info_logger.info("Features ALREADY transformed")

              transformed_data_path = self.config.root_dir
              train_data = np.load(os.path.join(transformed_data_path, 'Train.npz'))
              test_data = np.load(os.path.join(transformed_data_path, 'Test.npz'))

              X_train, y_train, groups_train = train_data["X_train"], train_data["y_train"], train_data["groups_train"]
              X_test, y_test = test_data["X_test"], test_data["y_test"]
              
              with open(self.config.STATUS_FILE,"w") as f:
                f.write(f"Feature engineering completed {True}")
              return X_train, X_test, y_train, y_test, groups_train
          else:
              return self.proceed_with_feature_engineering(X_train, X_test, y_train, y_test, groups_train)

      else:
          return self.proceed_with_feature_engineering(X_train, X_test, y_train, y_test, groups_train) 
    
      
    
    def save_transformed_data(self,X_train, X_test, y_train, y_test, groups_train):
        try:
            transformed_data_path = self.config.root_dir

            np.savez(os.path.join(transformed_data_path, 'Train.npz'), X_train=X_train, y_train=y_train, groups_train=groups_train)
            np.savez(os.path.join(transformed_data_path, 'Test.npz'),  X_test=X_test, y_test=y_test)

            with open(self.config.STATUS_FILE,"w") as f:
                f.write(f"Feature engineering completed {True}")
        except Exception as e:
            handle_exception(e, FeatureEngineeringError)
         