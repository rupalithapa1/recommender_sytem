import os
import sys
from joblib import dump
from joblib import load
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from training.exception import ModelEvaluationError,handle_exception
from training.custom_logging import info_logger, error_logger
from training.entity.config_entity import ModelEvaluationConfig
from training.configuration_manager.configuration import ConfigurationManager



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_test_data(self):
            
        try:
            info_logger.info("Loading the test data for Model Evaluation...")

            test_data_path = os.path.join(self.config.test_data_path,"Test.npz")

            # Loading the .npz files
            test_data = np.load(test_data_path)

            X_test, y_test = test_data["X_test"], test_data["y_test"]

            info_logger.info("Successfully loaded the test data for Model Evaluation...")

            return X_test, y_test
        except Exception as e:
            handle_exception(e, ModelEvaluationError)


    def load_final_model(self):
        try:
            final_model_path = self.config.model_path

            final_model = load(final_model_path)

            return final_model
        except Exception as e:
            handle_exception(e, ModelEvaluationError)
    
    def evaluate_final_model(self, final_model,X_test,y_test):
        try:
            info_logger.info("Evaluating final model...")

            y_pred = final_model.predict(X_test)

            report = classification_report(y_test,y_pred)

            with open(self.config.STATUS_FILE, "w") as f:
                f.write(report)

            # Generate classification report as a dictionary to save as metrics.json
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            metric_file_name = os.path.join(self.config.metric_file,"metrics.json")
            with open(metric_file_name, 'w') as f:
                json.dump(report_dict, f, indent=4)


            info_logger.info("Successfully evaluated the final model...")
        except Exception as e:
            handle_exception(e, ModelEvaluationError)