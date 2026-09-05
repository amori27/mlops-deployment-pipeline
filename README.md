# MLOps Deployment Pipeline

End-to-end ML lifecycle with Docker, MLflow, and GitHub Actions — trains, versions, serves, and monitors models with Kubernetes deployment configs.

## Usage

```bash
# Train a model
python src/training/train.py --config configs/train_config.yaml

# Serve with FastAPI
python src/serving/server.py --model_version latest

# Deploy to Kubernetes
kubectl apply -f kubernetes/deployment.yaml
```

## Stack

- **Training** — scikit-learn / XGBoost with MLflow experiment tracking
- **CI/CD** — GitHub Actions with model validation before deploy
- **Serving** — FastAPI with Docker containerization
- **Orchestration** — Kubernetes manifests + Helm charts
- **Monitoring** — Prometheus metrics endpoint for model drift detection

## License

MIT
