import yaml
from CreditRiskStructure.exception.exception import CreditRiskStructureException
from CreditRiskStructure.logging.logger import logging
import os, sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CreditRiskStructureException(e, sys)
    
def write_yaml_file(file_path: str, content: dict):
    """
    Writes a dictionary to a YAML file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file)
    except Exception as e:
        raise CreditRiskStructureException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.ndarray):
    """
    Saves a numpy array to a file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CreditRiskStructureException(e, sys)
    
def save_object(file_path: str, obj):
    """
    Saves a Python object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CreditRiskStructureException(e, sys)
    
def load_object(file_path: str):
    """
    Loads a Python object from a file using pickle.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CreditRiskStructureException(e, sys)
    
def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Loads a numpy array from a file.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CreditRiskStructureException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params) -> dict:
    """
    Evaluates a classification model using accuracy, precision, recall, and F1 score.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = roc_auc_score(y_train, y_train_pred)

            test_model_score = roc_auc_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CreditRiskStructureException(e, sys)