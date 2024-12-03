import os
import sys
from pathlib import Path
import shutil
from training.exception import DataIngestionError,handle_exception
from training.custom_logging import info_logger, error_logger
from training.entity.config_entity import DataIngestionConfig
from training.configuration_manager.configuration import ConfigurationManager

class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def save_data(self):
        try:
            status = None
            if not os.path.exists(self.config.data_dir):
                shutil.copytree(self.config.source, self.config.data_dir)
                status = True

                with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Data Ingestion status: {status}")

                info_logger.info(f"Data Ingestion status: {status}")

            else:
                if not os.listdir(self.config.data_dir):
                    info_logger.info(f"Folder '{self.config.data_dir}' already exists and is empty, proceeding with copy...")

                    shutil.copytree(self.config.source, self.config.data_dir)
                    status = True

                    with open(self.config.STATUS_FILE, "w") as f:
                            f.write(f"Data Ingestion status: {status}")

                    info_logger.info(f"Data Ingestion status: {status}")

                else:
                    info_logger.info(f"Folder '{self.config.data_dir}' already exists and is not empty, aborting operation.")
                    status = True
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Data Ingestion status: {status}")

        except Exception as e:
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Data Ingestion status: {status}")
            handle_exception(e, DataIngestionError)
        

# To test the component
if __name__ == "__main__":
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()

    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.save_data()