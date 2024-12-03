
from training.configuration_manager.configuration import ConfigurationManager
from training.components.final_train.feature_engineering import FeatureEngineering
from training.custom_logging import info_logger
import sys

PIPELINE = "Feature Engineering Training Pipeline"

class FeatureEngineeringTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        feature_engineering_config = config.get_feature_engineering_config()

        feature_engineering = FeatureEngineering(config=feature_engineering_config)
        
        # Transforming the data
        X_train, X_test, y_train, y_test, groups_train = feature_engineering.transform_features()

        # Saving the transformed data only if if does not exist
        feature_engineering.save_transformed_data(X_train, X_test, y_train, y_test, groups_train)
        
if __name__ == "__main__":
    info_logger.info(f">>>>> {PIPELINE} started <<<<")
    obj = FeatureEngineeringTrainingPipeline()
    obj.main()
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")

