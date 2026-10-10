import socket
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


class MLflowTracker:
    """Encapsulates experiment tracking and artifact logging with MLflow."""

    def __init__(self, config):
        self.config = config
        self.tracking_uri = config.mlflow_tracking_uri

        mlflow.set_tracking_uri(self.tracking_uri)

    def check_server(self, timeout = 3.0):
        """Check whether the MLflow server accepts connections."""
        parsed_uri = urlparse(self.tracking_uri)
        host = parsed_uri.hostname

        if not host:
            raise ValueError(
                f"Invalid MLflow tracking URI: {self.tracking_uri}"
            )

        default_port = 443 if parsed_uri.scheme == "https" else 80
        port = parsed_uri.port or default_port

        try:
            with socket.create_connection(
                (host, port),
                timeout=timeout,
            ):
                pass

        except OSError as exc:
            raise RuntimeError(
                f"MLflow server is unavailable at "
                f"{self.tracking_uri}. Start the server and try again."
            ) from exc

    def setup_experiment(self):
        """Verify connectivity and select the MLflow experiment."""
        self.check_server()
        mlflow.set_experiment(self.config.mlflow_experiment)

    def start_run(self, run_name):
        """Start an MLflow run as a context manager."""
        return mlflow.start_run(run_name=run_name)

    def log_parameters(self, parameters) -> None:
        """Log model parameters and configuration."""
        mlflow.log_params(parameters)

    def log_parameter(self, parameter_name, parameter_value):
        """Log a single model parameter and configuration."""
        mlflow.log_param(parameter_name, parameter_value)

    def log_metrics(self, dataset_name, metrics):
        """Log evaluation metrics for a dataset."""
        mlflow.log_metrics({
            f"{name}_{dataset_name}": float(value)
            for name, value in metrics.items()
        })

    def log_evaluation_artifacts(
        self,
        model,
        train_metrics,
        validation_metrics,
        test_metrics,
        Xval,
        yval,
        Xtest,
        ytest,
    ):
        """Save and log evaluation metrics and confusion matrices."""
        artifact_dir = self.config.mlflow_artifacts_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics as a CSV file.
        metrics_df = pd.DataFrame([
            {"dataset": "train", **train_metrics},
            {"dataset": "validation", **validation_metrics},
            {"dataset": "test", **test_metrics},
        ])

        metrics_path = artifact_dir / "metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation",
        )

        # Save confusion matrices for validation and test.
        datasets = [
            ("validation", Xval, yval),
            ("test", Xtest, ytest),
        ]

        for dataset_name, X, y in datasets:
            display = ConfusionMatrixDisplay.from_estimator(
                model,
                X,
                y,
                labels=["No", "Yes"],
                display_labels=["No", "Yes"],
                values_format="d",
            )

            display.ax_.set_title(
                f"Confusion Matrix - {dataset_name.capitalize()}"
            )

            figure_path = (
                artifact_dir / f"confusion_matrix_{dataset_name}.png"
            )

            try:
                display.figure_.tight_layout()
                display.figure_.savefig(
                    figure_path,
                    dpi=150,
                    bbox_inches="tight",
                )
                mlflow.log_artifact(
                    str(figure_path),
                    artifact_path="evaluation",
                )
            finally:
                plt.close(display.figure_)

    def log_model(self, model):
        """Register the trained pipeline as an MLflow model artifact."""
        mlflow.sklearn.log_model(
                                    sk_model=model,
                                    artifact_path="modelo_elastic_net",
                                    skops_trusted_types=["numpy.dtype"],
                                )