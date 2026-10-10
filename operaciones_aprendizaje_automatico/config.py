from dataclasses import dataclass
from pathlib import Path

#Current project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class ProjectConfig:
    raw_data_dir: Path = PROJECT_DIR / "data" / "raw"
    processed_data_dir: Path = PROJECT_DIR / "data" / "processed"
    reports_dir: Path = PROJECT_DIR / "reports"
    mlflow_artifacts_dir: Path = PROJECT_DIR / "reports" / "mlflow_artifacts"

    raw_filename: str = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    processed_filename: str = "processed_data.csv"

    target_column: str = "Attrition"

    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15

    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    #TC5061_Experimentacion_MLFlow_OrozcoJosue
    mlflow_experiment: str = "Experimentacion_MLFlow_Modelo_LogisticRegression_ElasticNet"

CONFIG = ProjectConfig()