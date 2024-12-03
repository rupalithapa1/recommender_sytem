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
class FeatureEngineeringConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    STATUS_FILE: str

#4
# Changes will be made as per the model is configured
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    random_search_models_nmf: Path
    model_cache_nmf:Path
    metric_file_name_nmf: Path
    best_model_params_nmf: Path
    STATUS_FILE: str
  
#5
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file: str
    STATUS_FILE: str


