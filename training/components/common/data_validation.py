import os
import pandas as pd
import datetime
from training.entity.config_entity import DataValidationConfig
from training.configuration_manager.configuration import ConfigurationManager
from training.exception import DataValidationError, handle_exception
from training.custom_logging import info_logger, error_logger

class DataValidation:
    def _init_(self, config: DataValidationConfig):
        self.config = config

    def validate_file_exists(self):
        """
        Validates if the source file exists.
        """
        if not os.path.exists(self.config.data_dir):
            raise FileNotFoundError(f"Source file not found at {self.config.data_dir}")

    def validate_columns(self, df: pd.DataFrame):
        """
        Validates if all required columns are present in the dataset.
        """
        required_columns = ['UserID','ItemID','Rating','PurchaseHistory','CartActivity','WishlistActivity',
                            'Clicks','Views','TimeSpentOnItem','PurchaseDate','SessionDuration',
                            'DeviceType','AbandonedCartData','Age','Gender','Location','Income',
                            'Occupation','SignUpDate','MembershipLevel','BrowsingHistory','Device'
                            'TimeOfInteraction','SearchQueries','ProductName','Category'
                            'Price','Discount','Brand','Description','Tags','Color','Size'
                            'Stock' 'Ratings','Reviews','ReleaseDate','PopularityScore']

       
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
info_logger.info(f"All required columns are present.")
def validate_missing_values(self, df: pd.DataFrame):
        """
        Validates if there are missing values in critical columns.
        """
        if df.isnull().sum().any():
            info_logger.warning("Dataset contains missing values.")
        else:
            info_logger.info("No missing values detected.")

def validate_data_types(self, df: pd.DataFrame):
        """
        Validates the data types of critical columns.
        """
        expected_types = {
             'ItemID'            :object,
             'Rating'            :int,     
             'PurchaseHistory'   : object,
             'CartActivity'       :  object,
             'WishlistActivity '  :   object,
             'Clicks '            :   int,
             'Views'              :   int,
             'TimeSpentOnItem '   :   float,
             'PurchaseDate '      :   datetime,
             'SessionDuration'    :   float,
             'DeviceType'         :   object,
             'AbandonedCartData'  :   object,
             'Age'                :   int,
             'Gender'             :   object,
             'Location'           :   object,
             'Income'            :   object,
             'Occupation'         :   object,
             'SignUpDate'         :   datetime,
            'MembershipLevel'    :   object,
            'BrowsingHistory'    :   object,
            'Device'             :   object,
            'TimeOfInteraction'  :   object,
            'SearchQueries'      :   object,
            'ProductName'        :   object,
            'Category'           :   object,
            'Price'              :   float,
            'Discount '          :   float,
            'Brand '             :   object,
            'Description'        :   object,
            'Tags'               :   object,
            'Color'              :   object,
            'Size'               :   object,
            'Stock'              :   int,
            'Ratings'            :   float,
            'Reviews'            :   object,
            'ReleaseDate'        :   datetime,
            'PopularityScore'    :   float,
            }

        mismatched_columns = []
        for column, expected_type in expected_types.items():
            if not pd.api.types.is_dtype_equal(df[column].dtype, expected_type):
                mismatched_columns.append(column)
        if mismatched_columns:
            raise TypeError(f"Data types mismatch for columns: {mismatched_columns}")
        info_logger.info("All data types match the expected schema.")
        def validate_data(self):
             """
        Orchestrates the data validation process.
        """
        try:
            # Check if the file exists
            self.validate_file_exists()

            # Load the dataset
            df = pd.read_csv(self.config.data_dir )
            info_logger.info(f"Loaded data from {self.config.data_dir} with shape {df.shape}.")

            # Perform validations
            self.validate_columns(df)
            self.validate_missing_values(df)
            self.validate_data_types(df)

            # Write validation status to the STATUS_FILE
            with open(self.config.STATUS_FILE, "w") as status_file:
                status_file.write("Data Validation completed: All checks passed successfully.")
            info_logger.info("Data Validation completed successfully.")

        except Exception as e:
            # Log and write failure status
            with open(self.config.STATUS_FILE, "w") as status_file:
                status_file.write(f"Data Validation failed due to an error: {str(e)}")
            handle_exception(e, DataValidationError)

# To check the component
if __name__ == "_main_":
    config_manager = ConfigurationManager()
    data_validation_config = config_manager.get_data_validation_config()

    data_validation = DataValidation(data_validation_config)
    data_validation.validate_data()