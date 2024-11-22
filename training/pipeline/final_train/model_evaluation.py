from training.configuration_manager.configuration import ConfigurationManager
from training.components.final_train.model_evaluation import ModelEvaluation
from training.custom_logging import info_logger
import sys

PIPELINE = "Final Model Evaluation Pipeline"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        #Load the data ingestion configuration object
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()

        model_evaluation = ModelEvaluation(config=model_evaluation_config)

        # Loading the test data for Model Evaluation
        X_test, y_test = model_evaluation.load_test_data()

        # Loading the final model
        final_model = model_evaluation.load_final_model()

        # Evluating the final_model
        model_evaluation.evaluate_final_model(final_model,X_test,y_test)

        


if __name__ == "__main__":  

        info_logger.info(f">>>>> {PIPELINE} started <<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")