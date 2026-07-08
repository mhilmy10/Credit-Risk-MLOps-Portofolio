## CREDIT RISK MLOps PORTOFOLIO

# Overview
An end-to-end MLOps project built to practice the full lifecycle of a machine learning
system as a Data Scientist — not just training a model, but productionizing it: pipeline
design, experiment tracking, serving, containerization, and CI/CD.

Goal of this project: demonstrate proper MLOps practices from raw data to a deployable,
reproducible service — the same workflow expected when handing a model off to an
engineering team in a real company.

# Features
- Training pipeline: data ingestion → validation → transformation → model training, with experiment tracking via MLflow/Dagshub
- Prediction API (FastAPI): single & batch (CSV) prediction endpoints
- Interactive UI (Streamlit): form-based single prediction with credit score gauge, plus batch CSV upload
- Containerized with Docker & orchestrated via Docker Compose
- CI/CD: GitHub Actions automatically builds & pushes images to Docker Hub on every push

# Tech Stack
Python, scikit-learn, FastAPI, Streamlit, MLflow, Docker, GitHub Actions

# Overview Model
This project using LogisticRegression with ROC-AUC perfomance 0.70

# How to run
    Run the Streamlit:
    docker compose up --build

    Retrain Model:
    python main.py

# Project Structure
├── .github/
│   └── workflows/
│       └── main.yaml           # CI/CD: build & push images to Docker Hub
├── app/                        # FastAPI application
│   ├── main.py
│   └── schemas.py
├── pipeline/                   # Batch prediction logic
│   └── batch_prediction.py
│   └── training_pipeline.py
├── CreditRiskStructure/        # Training pipeline components
│   ├── components/             # ingestion, validation, transformation, trainer
│   ├── utils/                  # shared utilities
│   ├── entity/                 # config & artifact dataclasses
│   └── constants/
├── data_schema/                # schema.yaml — expected column definitions
├── CreditRiskData/             # raw dataset (CSV)
├── final_model/                # trained model + preprocessor (synced)
├── notebooks/                  # Experimentation notebooks
├── streamlit_app.py            # Streamlit UI
├── main.py                     # Training pipeline entry point
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yaml
├── requirements.txt
└── setup.py
