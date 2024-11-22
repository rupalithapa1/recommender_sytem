from dataclasses import dataclass
from pathlib import Path 

#1
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source: Path
    data_dir: Path
    STATUS_FILE: str
#2
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_dir: Path
    STATUS_FILE: str

#3
@dataclass(frozen=True)
class ImageProcessingConfig:
    root_dir: Path
    data_dir: Path
    STATUS_FILE: str

#4
@dataclass(frozen=True)
class FeatureExtractionConfig:
    root_dir: Path
    data_dir: Path
    schema: dict
    STATUS_FILE: str

#5
@dataclass(frozen=True)
class FeatureEngineeringConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    STATUS_FILE: str

#6
# Changes will be made as per the model is configured
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    #best_cross_val_models_rf: Path
    final_model_name: str
    metric_file_name_rf: Path
    best_model_params_rf: Path
    STATUS_FILE: str
    #Hyperparameters
    #alpha: float
    #l1_ratio: float
    #target_column: str
#7
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    #all_params: dict
    metric_file: str
    #target_column: str
    #mlflow_uri: str
    STATUS_FILE: str


@dataclass(frozen=True)
class NestedCrossValConfig:
    root_dir: Path
    extracted_features: Path
    random_search_models_rf: Path
    model_cache_rf: Path
    #random_search_models_svm: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    STATUS_FILE: str
    metric_file_name_rf: Path
    best_model_params_rf: Path