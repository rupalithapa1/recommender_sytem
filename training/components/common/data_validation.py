import os
import sys
from concurrent.futures import ThreadPoolExecutor  # Allows multi-threading
from training.entity.config_entity import DataValidationConfig
from training.configuration_manager.configuration import ConfigurationManager
from training.exception import DataValidationError,handle_exception
from training.custom_logging import info_logger, error_logger



class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

        try:
            # Multithreading to speed up the validation process
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = list(executor.map(self.is_image_file, file_paths))

            # Check the results and log non-image files
            for file_path, is_image in zip(file_paths, results):
                if not is_image:
                    info_logger.info(f"Non-image file found: {file_path}")
                    all_images = False

            # Write the result to the STATUS_FILE
            status_message = f"Data Validation completed: {'All files are images' if all_images else 'Non-image files found'}"
            with open(self.config.STATUS_FILE, "w") as status_file:
                status_file.write(status_message)

            info_logger.info(status_message)

        except Exception as e:
            with open(self.config.STATUS_FILE, "w") as status_file:
                status_file.write(f"Data Validation failed due to an error: {str(e)}")
            handle_exception(e, DataValidationError)


# To check the component
if __name__ == "__main__":
    conifg = ConfigurationManager()
    data_validation_config = conifg.get_data_validation_config()

    data_validation = DataValidation(data_validation_config)
    data_validation.check_all_data_is_images()