"""Model Monitoring Module.

This module provides utilities for monitoring model performance,
drift detection, and alerting.
"""

import numpy as np
import pandas as pd
from typing import Any
from dataclasses import dataclass


@dataclass
class MonitoringMetrics:
    """Container for monitoring metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_drift: float
    feature_drift: float


class ModelMonitor:
    """Monitors model performance and detects drift."""

    def __init__(self, baseline_data: pd.DataFrame | None = None):
        """Initialize the ModelMonitor.

        Args:
            baseline_data: Baseline data for drift comparison.
        """
        self.baseline_data = baseline_data
        self.current_metrics: MonitoringMetrics | None = None
        self.history: list[MonitoringMetrics] = []

    def compute_drift(
        self,
        current_data: pd.DataFrame,
        baseline_data: pd.DataFrame | None = None
    ) -> dict[str, float]:
        """Compute drift between current and baseline data.

        Args:
            current_data: Current production data.
            baseline_data: Baseline data for comparison.

        Returns:
            Dictionary of drift metrics.
        """
        baseline = baseline_data or self.baseline_data
        if baseline is None:
            baseline = current_data

        drift_scores = {}

        for col in current_data.columns:
            if current_data[col].dtype in [np.float64, np.int64]:
                current_mean = current_data[col].mean()
                baseline_mean = baseline[col].mean()
                baseline_std = baseline[col].std()

                if baseline_std > 0:
                    drift = abs(current_mean - baseline_mean) / baseline_std
                    drift_scores[f"{col}_drift"] = float(drift)

        return drift_scores

    def check_drift_threshold(
        self,
        drift_scores: dict[str, float],
        threshold: float = 2.0
    ) -> list[str]:
        """Check if drift exceeds threshold.

        Args:
            drift_scores: Computed drift scores.
            threshold: Standard deviation threshold.

        Returns:
            List of features exceeding threshold.
        """
        drifted_features = []
        for feature, drift in drift_scores.items():
            if drift > threshold:
                drifted_features.append(feature)
        return drifted_features

    def log_metrics(
        self,
        metrics: MonitoringMetrics,
        timestamp: str | None = None
    ) -> None:
        """Log metrics to history.

        Args:
            metrics: Metrics to log.
            timestamp: Optional timestamp.
        """
        self.current_metrics = metrics
        self.history.append(metrics)

    def generate_report(self) -> dict[str, Any]:
        """Generate monitoring report.

        Returns:
            Dictionary containing monitoring report.
        """
        report = {
            "current_metrics": None,
            "historical_metrics": len(self.history),
            "drift_alerts": []
        }

        if self.current_metrics:
            report["current_metrics"] = {
                "accuracy": self.current_metrics.accuracy,
                "precision": self.current_metrics.precision,
                "recall": self.current_metrics.recall,
                "f1_score": self.current_metrics.f1_score
            }

        if len(self.history) > 1:
            latest = self.history[-1]
            previous = self.history[-2]

            if latest.accuracy < previous.accuracy - 0.05:
                report["drift_alerts"].append(
                    f"Accuracy drop detected: {previous.accuracy:.4f} -> {latest.accuracy:.4f}"
                )

        return report


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict[str, float]:
    """Calculate classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary of metrics.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted")),
        "recall": float(recall_score(y_true, y_pred, average="weighted")),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted"))
    }
