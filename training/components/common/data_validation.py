import os
import sys
from PIL import Image
from concurrent.futures import ThreadPoolExecutor  # Allows multi-threading
from training.entity.config_entity import DataValidationConfig
from training.configuration_manager.configuration import ConfigurationManager
from training.exception import DataValidationError,handle_exception
from training.custom_logging import info_logger, error_logger



# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    @staticmethod
    def is_image_file(file_path):
        """
        Checks if a file is a valid image by first checking the extension and then using the imghdr module 
        to verify the file type based on the file's content.
        
        Returns True if it's a valid image, False otherwise.
        """
        try:
            # Try to open the file as an image
            with Image.open(file_path) as img:
                img.verify()  # Verifies that the file contains valid image data
            return True
        except (IOError, SyntaxError):
            return False


    def check_all_data_is_images(self):
        """
        Checks whether all files in the folder and its subfolders are valid images using multithreading.
        Logs the process and writes the status to the STATUS_FILE.
        """
        all_images = True
        num_threads = 8  # Number of threads to use for concurrent file validation.
        file_paths = []

        try:
            # Walk through the directory tree to collect all file paths
            for root, dirs, files in os.walk(self.config.data_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)

            # Log the start of image validation
            info_logger.info(f"Starting image validation for {len(file_paths)} files in {self.config.data_dir}")

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