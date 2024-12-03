from training.configuration_manager.configuration import ConfigurationManager
from training.components.final_train.model_training import ModelTraining
from training.custom_logging import info_logger
import sys

PIPELINE = "Final Model Training Pipeline"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        #Load the data ingestion configuration object
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()

        model_trainer = ModelTraining(config=model_trainer_config)

        # Loading the train data and test data for Final Training
        X_train, X_test, y_train,y_test, groups_train = model_trainer.load_transformed_data()

        # Load the naem of best of the models of NestedCrossVAlidation (This is actually the count of the model which is the best)
        best_model_count = model_trainer.load_best_model() 

        # Training the final model for Final Training
        final_model = model_trainer.train_final_model(best_model_count, X_train, y_train)

        # Save the final model for Final Training
        model_trainer.save_final_model(final_model)
        
if __name__ == "__main__":

        info_logger.info(f">>>>> {PIPELINE} started <<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")