import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from CreditRiskStructure.exception.exception import CreditRiskStructureException
from CreditRiskStructure.logging.logger import logging
from CreditRiskStructure.entity.config_entity import DataTransformationConfig
from CreditRiskStructure.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from CreditRiskStructure.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS, TARGET_COLUMN
from CreditRiskStructure.utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_validation_artifact: DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
    
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
        
    def get_data_transformation_object(self, df: pd.DataFrame) -> Pipeline:
        try:
            num_features = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_features = df.select_dtypes(include=[object]).columns.tolist()
            # Create a pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ('imputer', KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS))
            ])

            # Create a pipeline for categorical features
            cat_pipeline = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine the numerical and categorical pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, num_features),
                ('cat', cat_pipeline, cat_features)
            ])

            return preprocessor
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Starting data transformation process")
        try:
            # train and test data read from valid data path
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)
            # move target column to the end of the dataframe
            train_df = train_df[[col for col in train_df.columns if col != TARGET_COLUMN] + [TARGET_COLUMN]]
            test_df = test_df[[col for col in test_df.columns if col != TARGET_COLUMN] + [TARGET_COLUMN]]
            # drop target column from train and test data
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            # target column from train and test data
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]
            # get data transformation object
            preprocessor = self.get_data_transformation_object(input_feature_train_df)
            # fit and transform on train data and transform on test data
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            # concatenate the target column back
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            # save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            # save preprocessor object 
            save_object(self.data_transformation_config.object_file_path, preprocessor)
            save_object("final_model/preprocessor.pkl", preprocessor)
            # prepare artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.object_file_path
            )
            logging.info("Data transformation process completed successfully")
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CreditRiskStructureException(e, sys)