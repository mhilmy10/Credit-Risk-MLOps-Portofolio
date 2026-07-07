"""
Pydantic schemas for the Credit Risk Prediction API.

These mirror the columns defined in data_schema/schema.yaml
(minus the target column `loan_status`, which is what we predict).
"""

from pydantic import BaseModel, Field
from typing import List, Literal


class LoanApplicant(BaseModel):
    person_age: int = Field(..., ge=18, le=100, description="Age of the applicant")
    person_income: float = Field(..., ge=0, description="Annual income")
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: float = Field(..., ge=0, description="Employment length in years")
    loan_intent: Literal[
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
        "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(..., ge=0, description="Loan amount requested")
    loan_int_rate: float = Field(..., ge=0, description="Loan interest rate (%)")
    loan_percent_income: float = Field(..., ge=0, description="Loan amount as a percentage of income")
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(..., ge=0, description="Credit history length in years")

    class Config:
        json_schema_extra = {
            "example": {
                "person_age": 25,
                "person_income": 55000,
                "person_home_ownership": "RENT",
                "person_emp_length": 3.0,
                "loan_intent": "EDUCATION",
                "loan_grade": "B",
                "loan_amnt": 10000,
                "loan_int_rate": 11.5,
                "loan_percent_income": 0.18,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 4
            }
        }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = Low Risk (loan likely not default), 1 = High Risk (likely default)")
    label: int = Field(..., description="Human readable label")
    probability_default: float = Field(..., description="Predicted probability of default")
    risk_score: int = Field(..., description="Default Risk Score, High Score means low risk")


class BatchPredictionItem(PredictionResponse):
    row_index: int


class BatchPredictionResponse(BaseModel):
    total_records: int
    results: List[BatchPredictionItem]
