import mlflow
import mlflow.sklearn

from .model import build_model
from .evaluate import evaluate_model


class ModelTrainer:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.model = None


    def fit(self, X_train, y_train):
        self.model = build_model(self.args)
        self.model.fit(X_train, y_train)
        return self.model


    def evaluate(self, X, y):
        if self.model is None:
            raise RuntimeError(
                "El modelo debe entrenarse antes de evaluarse."
            )

        return evaluate_model(self.model, X, y)


    def log_metrics(self, dataset_name, metrics):
        for metric_name, value in metrics.items():
            mlflow.log_metric(
                f"{metric_name}_{dataset_name}",
                float(value),
            )

    def log_model(self):
        if self.model is None:
            raise RuntimeError(
                "No hay un modelo entrenado para registrar."
            )

        mlflow.sklearn.log_model(
            self.model,
            name="modelo_elastic_net",
        )
