from CreditRiskStructure.logging.logger import logging
from CreditRiskStructure.exception.exception import CreditRiskStructureException
from CreditRiskStructure.entity.config_entity import DataIngestionConfig
from CreditRiskStructure.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pymongo
from typing import List
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
        
    def export_collection_as_dataframe(self):
        """
        Import Data from MongoDB collection as DataFrame
        """
        try:
            self.database_name = self.data_ingestion_config.data_ingestion_database_name
            self.collection_name = self.data_ingestion_config.data_ingestion_collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[self.database_name][self.collection_name]
            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df=df.drop("_id",axis=1)
            
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise CreditRiskStructureException(e, sys)

    def export_data_into_feature_store(self, df: pd.DataFrame):
        """
        Export Data into Feature Store and Saved as CSV file
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_dir
            # creating folder
            logging.info(f"Creating directory at {feature_store_file_path}")
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            df.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Data exported to feature store at {feature_store_file_path}")
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
        
    def split_data_as_train_test(self, df: pd.DataFrame):
        """
        Split Data into Train and Test and Save as CSV file
        """
        try:
            train_set, test_set = train_test_split(df, 
                                                   test_size=self.data_ingestion_config.train_test_split_ratio, 
                                                   random_state=42)
            logging.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")
            # creating folder
            logging.info(f"Creating train test directory at {self.data_ingestion_config.ingested_dir}")
            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)
            logging.info(f"Train and Test data saved at {self.data_ingestion_config.train_file_path} and {self.data_ingestion_config.test_file_path}")
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
        
    def initiate_data_ingestion(self):
        """
        Initiate Data Ingestion Process
        """
        try:
            df = self.export_collection_as_dataframe()
            self.export_data_into_feature_store(df)
            self.split_data_as_train_test(df)
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path)
            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CreditRiskStructureException(e, sys)