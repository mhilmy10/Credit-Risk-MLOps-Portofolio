from CreditRiskStructure.logging.logger import logging
from CreditRiskStructure.exception.exception import CreditRiskStructureException
from CreditRiskStructure.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from CreditRiskStructure.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from CreditRiskStructure.components.data_ingestion import DataIngestion
from CreditRiskStructure.components.data_validation import DataValidation
from CreditRiskStructure.components.data_transformation import DataTransformation
from CreditRiskStructure.components.model_trainer import ModelTrainer
import os  
import sys

if __name__ == "__main__":
    try:
        # data ingestion process
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)    
        logging.info("Iniating Data Ingestion Process")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        logging.info(f"Data Ingestion Artifact Completed")
        # data validation process
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_validation_config, data_ingestion_artifact)
        logging.info("Iniating Data Validation Process")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Data Validation Artifact Completed")
        print(data_validation_artifact)
        #data transformation process
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config, data_validation_artifact)
        logging.info("Iniating Data Transformation Process")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(f"Data Transformation Artifact Completed")
        print(data_transformation_artifact)
        # model trainer process
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        logging.info("Iniating Model Trainer Process")
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info(f"Model Trainer Artifact Completed")
        print(model_trainer_artifact)
    except Exception as e:
        raise CreditRiskStructureException(e, sys)