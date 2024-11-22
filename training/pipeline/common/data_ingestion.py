from training.configuration_manager.configuration import ConfigurationManager
from training.components.common.data_ingestion import DataIngestion
from training.custom_logging import info_logger
import sys

PIPELINE = "Data Ingestion Training Pipeline"
class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        #Load the data ingestion configuration object
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()

        # Pass the data ingestion configuration obj to the Data Ingestion component
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.save_data()
        


    

if __name__ == "__main__":
    info_logger.info(f">>>>> {PIPELINE} started <<<<")
    obj = DataIngestionPipeline()
    obj.main()
    info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")