import sys
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from CreditRiskStructure.exception.exception import CreditRiskStructureException
from CreditRiskStructure.logging.logger import logging
from CreditRiskStructure.utils.ml_utils.scoring import probability_to_score
from pipeline.batch_prediction import BatchPredictionPipeline
from app.schemas import LoanApplicant, PredictionResponse, BatchPredictionResponse, BatchPredictionItem


app = FastAPI(
    title="Credit Risk Prediction API",
    description="Serves the trained Credit Risk model (Logistic Regression) for single and batch predictions.",
    version="1.0.0"
)

# Allow the Streamlit frontend (running on a different host/port/container) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, reused across all requests
pipeline: BatchPredictionPipeline | None = None


@app.on_event("startup")
def startup_event():
    global pipeline
    try:
        pipeline = BatchPredictionPipeline()
        logging.info("BatchPredictionPipeline loaded successfully at startup")
    except Exception as e:
        logging.error(f"Failed to load model at startup: {e}")
        pipeline = None


@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(applicant: LoanApplicant):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check server logs.")
    try:
        df = pd.DataFrame([applicant.model_dump()])
        prediction = pipeline.risk_model.predict(df)
        probability = pipeline.risk_model.predict_proba(df)

        pred_value = int(prediction[0])
        prob_value = float(probability[0]) if probability is not None else 0.0

        score = probability_to_score(prob_value)

        return PredictionResponse(
            prediction=pred_value,
            label=pred_value,
            probability_default=round(prob_value, 4),
            risk_score=score
        )
    except CreditRiskStructureException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check server logs.")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")
    try:
        content = file.file.read()
        df = pd.read_csv(io.BytesIO(content))
        pipeline.validate_columns(df)  # raises if columns are missing

        result_df = pipeline.run_batch_prediction(df)

        items = [
            BatchPredictionItem(
                row_index=idx,
                prediction=int(row["prediction"]),
                label=row["label"],
                probability_default=round(float(row["probability_default"]), 4),
                risk_score=int(row["scores"]))
            for idx, row in result_df.iterrows()
        ]

        return BatchPredictionResponse(total_records=len(items), results=items)
    except CreditRiskStructureException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch-csv", tags=["Prediction"])
def predict_batch_csv(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check server logs.")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")
    try:
        content = file.file.read()
        df = pd.read_csv(io.BytesIO(content))
        pipeline.validate_columns(df)

        result_df = pipeline.run_batch_prediction(df)

        stream = io.StringIO()
        result_df.to_csv(stream, index=False)
        stream.seek(0)

        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=prediction_results.csv"}
        )
    except CreditRiskStructureException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
