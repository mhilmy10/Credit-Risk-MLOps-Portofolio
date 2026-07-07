import os
import sys
import pandas as pd
from CreditRiskStructure.exception.exception import CreditRiskStructureException
from CreditRiskStructure.logging.logger import logging
from CreditRiskStructure.utils.utils import load_object
from CreditRiskStructure.utils.ml_utils.scoring import probability_to_score



class BatchPredictionPipeline:
    expected_columns = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"
]
    def __init__(self, model_path: str = os.path.join("final_model", "risk_model.pkl")):
        try:
            self.model_path = model_path
            self.risk_model = self._load_risk_model()
        except Exception as e:
            raise CreditRiskStructureException(e, sys)

    def _load_risk_model(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            return load_object(self.model_path)
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
    
    def validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            missing = [col for col in self.expected_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            return df[self.expected_columns]
        except Exception as e:
            raise CreditRiskStructureException(e, sys)

    def run_batch_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Starting batch prediction")

            clean_df = self.validate_columns(df.copy())

            predictions = self.risk_model.predict(clean_df)
            probabilities = self.risk_model.predict_proba(clean_df)

            result_df = df.copy()
            result_df["prediction"] = predictions.astype(int)
            result_df["label"] = result_df["prediction"]
            result_df["scores"] = probability_to_score(probabilities)
            result_df["probability_default"] = probabilities if probabilities is not None else None

            logging.info(f"Batch prediction completed for {len(result_df)} records")
            return result_df
        except Exception as e:
            raise CreditRiskStructureException(e, sys)

    

