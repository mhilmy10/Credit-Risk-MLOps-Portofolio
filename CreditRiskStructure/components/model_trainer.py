from CreditRiskStructure.entity.config_entity import ModelTrainerConfig
from CreditRiskStructure.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from CreditRiskStructure.exception.exception import CreditRiskStructureException
from CreditRiskStructure.logging.logger import logging
from CreditRiskStructure.utils.utils import load_object, save_object, load_numpy_array_data, evaluate_model
from CreditRiskStructure.utils.ml_utils.model.estimator import CreditRiskModel
import sys
from CreditRiskStructure.utils.ml_utils.metric.classification_metric import get_classification_score
import os
import mlflow

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost
from xgboost import XGBClassifier
from urllib.parse import urlparse
import dagshub
dagshub.init(repo_owner='mhilmy10', repo_name='Credit-Risk-MLOps-Portofolio', mlflow=True)

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/mhilmy10/Credit-Risk-MLOps-Portofolio.mlflow"
#mlflow.register_model(model_uri="https://dagshub.com/mhilmy10/Credit-Risk-MLOps-Portofolio.mlflow"
#                      ,name="CreditRiskModel")


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
        
    def track_mlflow(self, best_model, classification_metric):
        mlflow.set_registry_uri("https://dagshub.com/mhilmy10/Credit-Risk-MLOps-Portofolio.mlflow")
        tracking_uri_scheme = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run() as mlflow_run:
            f1_score = classification_metric.f1_score
            precision_score = classification_metric.precision_score
            recall_score = classification_metric.recall_score
            roc_auc = classification_metric.roc_auc
            gini = classification_metric.gini

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision_score", precision_score)
            mlflow.log_metric("recall_score", recall_score)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("gini", gini)
            mlflow.sklearn.log_model(best_model, "model")
            if tracking_uri_scheme != "file":
                mlflow.sklearn.log_model(best_model, "model", registered_model_name="CreditRiskModel")
            else:
                logging.info("Since the tracking uri is not a file store, model is not registered.")
        
    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "LogisticRegression": LogisticRegression()}
            params = {
                "LogisticRegression": {
                    'max_iter': [100, 200, 300],
                }
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models, params)
            
            # get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))
            
            # get the best model name from the dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # get the classification metrics for train and test data
            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_train, y_train_pred)
            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_test, y_test_pred)
            logging.info(f"Best Model Found, Model Name: {best_model_name}, Accuracy Score: {best_model_score}, Classification Train Metric: {classification_train_metric}, Classification Test Metric: {classification_test_metric}")
            
            # MLflow Track
            self.track_mlflow(best_model, classification_train_metric)

            # save the best model and preprocessor object
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            risk_model = CreditRiskModel(preprocessor=preprocessor, model=best_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=CreditRiskModel)
            save_object("final_model/risk_model.pkl", risk_model)
            
            # create the model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric)
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CreditRiskStructureException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            # Load transformed training and testing data
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            train_array = load_numpy_array_data(transformed_train_file_path)
            test_array = load_numpy_array_data(transformed_test_file_path)
            # Split the data into features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            # train the model and get the model trainer artifact
            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact
        except Exception as e:
            raise CreditRiskStructureException(e, sys)