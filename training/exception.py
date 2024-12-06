from training.custom_logging import error_logger

# Custom exception handler function
def handle_exception(error, error_type):
    """
    Logs the error with details to the log file and prints a formatted message to the console.
    
    Args:
        error (Exception): The original exception instance.
        error_type (PipelineError): The specific pipeline error type (e.g., DataIngestionError).
    """
    # Log the complete stack trace and details to the log file
    error_logger.error("Exception occurred", exc_info=True)

    # Add blank lines to separate error messages in the log
    error_logger.error("\n\n")  

    # Print only the formatted message to the console
    print(f"{error_type.__name__}: {error.__class__.__name__}: {error}")

# Base custom exception
class PipelineError(Exception):
    """Base class for custom pipeline exceptions"""
    def __init__(self, original_exception):
        super().__init__(str(original_exception))
        self.original_exception = original_exception

# Specific pipeline exceptions
class DataIngestionError(PipelineError):
    pass

class DataValidationError(PipelineError):
    pass

class FeatureEngineeringError(PipelineError):
    pass

class ModelTrainingError(PipelineError):
    pass

class ModelEvaluationError(PipelineError):
    pass

# Example function to simulate an error for testing
def my_func(value):
    """
    A simple example function that raises an exception.
    
    Args:
        value (int): An integer input.
    """
    if value != 42:
        raise ValueError("Value must be 42!")

# Main script
if __name__ == "__main__":
    try:
        # Call a function that may raise an exception
        my_func(23)  # This will raise a ValueError
    except ValueError as e:
        # Handle the exception and wrap it in a custom pipeline error
        handle_exception(e, DataIngestionError)
