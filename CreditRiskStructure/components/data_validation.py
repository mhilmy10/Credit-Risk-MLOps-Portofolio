from CreditRiskStructure.logging.logger import logging
from CreditRiskStructure.exception.exception import CreditRiskStructureException
from CreditRiskStructure.entity.config_entity import DataValidationConfig
from CreditRiskStructure.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from CreditRiskStructure.utils.utils import read_yaml_file, write_yaml_file
from CreditRiskStructure.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import os
import sys
import pandas as pd

class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
    
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
        
    @staticmethod
    def data_cleansing(df: pd.DataFrame):
        try:
            before = len(df)
            missing_values_before = df.isna().sum()
            logging.info(f"Data Frame Shape Before Cleansing : {before}")
            logging.info(f"Missing Values :{missing_values_before}")
            df = df.copy()
            df = df.drop_duplicates()
            df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())
            df = df.drop(df[df['person_emp_length'].isin([123, 124])].index)
            missing_values_after = df.isna().sum()
            after = len(df)
            logging.info(f"Duplicate removed : {before-after}")
            logging.info(f"Missing Values After Cleansing : {missing_values_after}")
            return df
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
        
    def validate_data_schema(self, df: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self.schema_config["columns"])
            logging.info(f"Expected number of columns: {number_of_columns}")
            logging.info(f"Actual number of columns: {len(df.columns)}")
            if number_of_columns == len(df.columns):
                return True
            return False
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
                
    def detect_data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold = 0.5) -> bool:
        try:
            status = True
            drift_report = {}
            numerical_columns = base_df.select_dtypes(include=['int64', 'float64']).columns
            for column in numerical_columns:
                base_data = base_df[column]
                current_data = current_df[column]
                is_same_dist = ks_2samp(base_data, current_data)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                drift_report.update({column: {
                    "p_value": float(is_same_dist.pvalue), 
                    "drift_status": is_found}})
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            # create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=drift_report)
            return status
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            # data reading
            logging.info("Reading training and testing data")
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)
            # data cleansing
            logging.info("Data Cleansing Process")
            train_df = DataValidation.data_cleansing(train_df)
            test_df = DataValidation.data_cleansing(test_df)
            # data validation process
            logging.info("Validating training and testing data schema")
            is_train_validated = self.validate_data_schema(train_df)
            is_test_validated = self.validate_data_schema(test_df)
            if not is_train_validated:
                raise Exception("Training data schema validation failed")
            if not is_test_validated:
                raise Exception("Testing data schema validation failed")
            # data drift detection
            logging.info("Detecting data drift between training and testing data")
            drift_status = self.detect_data_drift(base_df=train_df, current_df=test_df)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)
            logging.info("Data validation completed successfully")
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifact
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
