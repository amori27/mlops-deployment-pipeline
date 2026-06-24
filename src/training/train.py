"""Model Training Module.

This module handles model training with MLflow tracking,
hyperparameter tuning, and model registration.
"""

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification
from typing import Any


class ModelTrainer:
    """Handles model training with experiment tracking."""

    def __init__(self, experiment_name: str = "default"):
        """Initialize the ModelTrainer.

        Args:
            experiment_name: MLflow experiment name.
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.model = None
        self.metrics = {}

    def prepare_data(self, n_samples: int = 1000, n_features: int = 20) -> tuple:
        """Prepare synthetic training data.

        Args:
            n_samples: Number of samples to generate.
            n_features: Number of features.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,
            n_redundant=5,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: dict[str, Any] | None = None
    ) -> RandomForestClassifier:
        """Train a model with MLflow tracking.

        Args:
            X_train: Training features.
            y_train: Training labels.
            params: Model hyperparameters.

        Returns:
            Trained model.
        """
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42
        }
        params = params or default_params

        with mlflow.start_run(run_name="model_training"):
            mlflow.log_params(params)

            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            train_score = model.score(X_train, y_train)
            mlflow.log_metric("train_accuracy", train_score)

            mlflow.sklearn.log_model(model, "model")

            self.model = model
            return model

    def hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: dict[str, list]
    ) -> dict[str, Any]:
        """Perform hyperparameter tuning with cross-validation.

        Args:
            X_train: Training features.
            y_train: Training labels.
            param_grid: Grid of parameters to search.

        Returns:
            Best parameters and scores.
        """
        with mlflow.start_run(run_name="hyperparameter_tuning"):
            base_model = RandomForestClassifier(random_state=42)

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)

            return {
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "best_model": grid_search.best_estimator_
            }

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict[str, float]:
        """Evaluate model on test data.

        Args:
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        test_score = self.model.score(X_test, y_test)
        self.metrics["test_accuracy"] = test_score

        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metric("test_accuracy", test_score)

        return self.metrics


def train_production_model(
    data_path: str | None = None,
    experiment_name: str = "production"
) -> RandomForestClassifier:
    """Train a production model with full pipeline.

    Args:
        data_path: Optional path to training data.
        experiment_name: MLflow experiment name.

    Returns:
        Trained model ready for deployment.
    """
    trainer = ModelTrainer(experiment_name=experiment_name)

    X_train, X_test, y_train, y_test = trainer.prepare_data(n_samples=2000)

    model = trainer.train(X_train, y_train)

    metrics = trainer.evaluate(X_test, y_test)

    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model",
        "ProductionModel"
    )

    return model


if __name__ == "__main__":
    trainer = ModelTrainer("quick_test")
    X_train, X_test, y_train, y_test = trainer.prepare_data()
    model = trainer.train(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)
    print(f"Training complete. Test accuracy: {metrics['test_accuracy']:.4f}")
