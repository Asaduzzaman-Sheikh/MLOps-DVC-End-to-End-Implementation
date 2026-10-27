import os
import pandas as pd
import logging
import numpy as np
import pickle
import json

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score


# ------------------------------------------------------ LOGGING SETUP ------------------------------------------------------ #
log_dir = 'logs'
os.makedirs(log_dir, exist_ok = True)

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ------------------------------------------------------ MODEL EVALUATION STAGE ------------------------------------------------------ #
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and return it as a pandas DataFrame.
    
    Args:
        file_path (str): The path to the CSV file to be loaded.
        
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}. Please check the file path.")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"No data: The file at {file_path} is empty.")
        raise
    except pd.errors.ParserError:
        logger.error(f"Parsing error: The file at {file_path} could not be parsed.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data from {file_path}: {e}")
        raise


def load_model(model_path: str):
    """
    Load a trained machine learning model from a file.
    
    Args:
        model_path (str): The path to the file containing the trained model.
    
    Returns:
        The loaded machine learning model.
    """
    try: 
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            logger.debug(f"Model loaded successfully from {model_path}")
            return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}. Please check the file path.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the model from {model_path}: {e}")
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the trained model on the test dataset and return evaluation metrics.
    
    Args:
        model: The trained machine learning model.
        X_test (np.ndarray): The test features.
        y_test (np.ndarray): The true labels for the test data.
        
    Returns:
        dict: A dictionary containing evaluation metrics (accuracy, precision, recall, F1 score, and ROC AUC score).
    """
    try:
        # Make predictions on the test data
        yhat = model.predict(X_test)
        yhat_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, yhat)
        precision = precision_score(y_test, yhat, average='binary', zero_division=0)
        recall = recall_score(y_test, yhat, average='binary', zero_division=0)
        f1 = f1_score(y_test, yhat, average='binary', zero_division=0)
        roc_auc = roc_auc_score(y_test, yhat_proba) if yhat_proba is not None else None
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc_score": roc_auc
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"An error occurred during model evaluation: {e}")
        raise

# Save evaluation metrics to a JSON file
def save_metrics(metrics: dict, report_dir: str) -> None:
    """
    Save evaluation metrics to a JSON file in the specified report directory.
    
    Args:
        metrics (dict): A dictionary containing evaluation metrics.
        report_dir (str): The directory where the metrics report will be saved.
    """
    try:
        os.makedirs(report_dir, exist_ok = True)
        report_path = os.path.join(report_dir, 'evaluation_metrics.json')
        
        with open(report_path, 'w') as file:
            json.dump(metrics, file, indent = 4)
            logger.debug(f"Evaluation metrics saved successfully at {report_path}")
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error while saving evaluation metrics: {fnf_error}")
        raise
    except PermissionError as perm_error:
        logger.error(f"Permission error while saving evaluation metrics: {perm_error}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while saving evaluation metrics: {e}")
        raise
# ------------------------------------------------------ MAIN FUNCTION   ---------------------------------------------------------
def main():
    """
    Main function to evaluate the trained model on the test dataset.
    """
    try:
        # Load the test dataset
        test_data = load_data('./data/processed/test_transformed.csv')
        
        X_test = test_data.iloc[:, :-1].values
        y_test =test_data.iloc[:, -1].values
        
        # Load the model
        model = load_model('./models/trained_model.pkl')
        logger.debug("Model loaded successfully for evaluation.")
        
        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)
        logger.info(f"Model evaluation completed successfully.")
        
        # Save the evaluation metrics to a JSON file
        report_dir = './reports'
        save_metrics(metrics, report_dir)
        logger.info(f"Evaluation metrics saved successfully at {report_dir}")
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error: {fnf_error}")
        raise
    except Exception as e:
        logger.error(f"An error occurred during model evaluation: {e}")
        
if __name__ == "__main__":
    main()

