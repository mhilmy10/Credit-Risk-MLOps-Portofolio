import os  
import sys
from CreditRiskStructure.logging.logger import logging
from CreditRiskStructure.exception.exception import CreditRiskStructureException
from CreditRiskStructure.entity.config_entity import (
    DataIngestionConfig, 
    TrainingPipelineConfig, 
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig)
from CreditRiskStructure.entity.artifact_entity import (
    DataIngestionArtifact, 
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact)
from CreditRiskStructure.components.data_ingestion import DataIngestion
from CreditRiskStructure.components.data_validation import DataValidation
from CreditRiskStructure.components.data_transformation import DataTransformation
from CreditRiskStructure.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.training_pipeline_config = training_pipeline_config
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion process")
            data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion process completed")
            return data_ingestion_artifact
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation process")
            data_validation_config = DataValidationConfig(self.training_pipeline_config)
            data_validation = DataValidation(data_validation_config, data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation process completed")
            return data_validation_artifact
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation process")
            data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            data_transformation = DataTransformation(data_transformation_config, data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation process completed")
            return data_transformation_artifact
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model trainer process")
            model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model trainer process completed")
            return model_trainer_artifact
        except Exception as e:
            raise CreditRiskStructureException(e, sys)
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            logging.info("Training pipeline completed successfully")
        except Exception as e:
            raise CreditRiskStructureException(e, sys)