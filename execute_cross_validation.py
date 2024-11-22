from training.pipeline.common.data_ingestion import DataIngestionPipeline
from training.pipeline.common.data_validation import DataValidationPipeline
from training.pipeline.common.feature_extraction import FeatureExtractionPipeline
from training.pipeline.cross_val.nested_cross_val import NestedCrossValPipeline

from training.custom_logging import info_logger, error_logger
import sys
import os



PIPELINE = "Data Ingestion Training Pipeline"
info_logger.info(f">>>>> {PIPELINE} started <<<<")
data_ingestion = DataIngestionPipeline()
data_ingestion.main()
info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")



PIPELINE = "Data Validation Training Pipeline"
info_logger.info(f">>>>> {PIPELINE} started <<<<")
data_validation = DataValidationPipeline()
data_validation.main()
info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")


PIPELINE = "Feature Extraction Training Pipeline"
info_logger.info(f">>>>> {PIPELINE} started <<<<")
feature_extraction = FeatureExtractionPipeline()
feature_extraction.main()
info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")


PIPELINE = "Nested Cross ValidationTraining Pipeline"
info_logger.info(f">>>>> {PIPELINE} started <<<<")
nested_cross_val = NestedCrossValPipeline()
nested_cross_val.main()
info_logger.info(f">>>>>>>> {PIPELINE} completed <<<<<<<<<")