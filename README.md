# MLOps Deployment Pipeline

A complete MLOps pipeline for automated model training, versioning, deployment, and monitoring using Docker, CI/CD, and model registry.

## Description

Production-ready MLOps pipeline that handles the complete ML lifecycle from data ingestion to model deployment. Features include Docker containerization, GitHub Actions CI/CD, MLflow model versioning, and Kubernetes deployment configurations.

## Skills & Technologies

- Docker & Kubernetes
- GitHub Actions CI/CD
- MLflow for Model Versioning
- Python 3.9+
- FastAPI for Model Serving
- Kubernetes Manifests
- Helm Charts

## Installation

```bash
git clone https://github.com/AmirAsaad/mlops-deployment-pipeline.git
cd mlops-deployment-pipeline
pip install -r requirements.txt
docker build -t ml-pipeline:latest .
```

## Usage

### Training Pipeline

```bash
python src/training/train.py --config configs/train_config.yaml
```

### Model Serving

```bash
python src/serving/server.py --model_version latest
```

### Kubernetes Deployment

```bash
kubectl apply -f kubernetes/deployment.yaml
```

## Project Structure

```
mlops-deployment-pipeline/
├── src/
│   ├── training/           # Model training scripts
│   ├── serving/           # Model serving API
│   └── monitoring/         # Model monitoring
├── kubernetes/             # K8s manifests
├── docker/                 # Dockerfile configs
├── github/workflows/       # CI/CD pipelines
├── configs/               # Configuration files
├── requirements.txt
└── README.md
```

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Docker Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [GitHub Actions](https://docs.github.com/en/actions)

## License

MIT License
