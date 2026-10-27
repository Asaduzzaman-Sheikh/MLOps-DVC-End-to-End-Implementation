import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import logging

import pickle

# ------------------------------------------------------ LOGGING SETUP ------------------------------------------------------ #
log_dir = 'logs'
os.makedirs(log_dir, exist_ok = True)

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ------------------------------------------------------ MODEL BUILDING STAGE ------------------------------------------------------ #
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
    
def build_model(X_train: np.ndarray, y_train: np.ndarray, params: dict)-> object:
    """
    Build and train a machine learning model using the provided training data.
    
    Args:
        X_train (np.ndarray): The training features.
        y_train (np.ndarray): The training labels.
        params (dict): The parameters for the model.
    Returns:
        object: The trained machine learning model.
    """
    try:
        # Initialize the model
        model = SVC(**params)
        
        # Train the model
        model.fit(X_train, y_train)
        logger.debug("Model trained successfully.")
        
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
    
# Define a function to save the trained model
def save_model(model: object, model_dir: str) -> None:
    """
    Save the trained model to a specified directory.
    
    Args:
        model (object): The trained machine learning model.
        model_dir (str): The directory where the model will be saved.
    """
    try:
        os.makedirs(model_dir, exist_ok = True)
        model_path = os.path.join(model_dir, 'trained_model.pkl')
        
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
            logger.debug(f"Model saved successfully at {model_path}")
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error while saving the model: {fnf_error}")
        raise
    except PermissionError as perm_error:
        logger.error(f"Permission error while saving the model: {perm_error}")
        raise
    except Exception as e:
        logger.error(f"Error while saving the model: {e}")
        raise
# ---------------------------------------------------------- MAIN FUNCTION ---------------------------------------------------------- #
def main():
    try:
        # Load the training and testing data
        train_data = load_data('./data/processed/train_transformed.csv')
        # test_data = load_data('./data/processed/test_transformed.csv')
        logger.debug("Training data loaded successfully.")
        
        # Start the model building process
        logger.info("Starting model building process...")
        params = {
            'kernel': 'poly',
            'degree': 3,
            'class_weight': 'balanced',
            'gamma': 'scale',
            'probability': True 
        }
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        
        
        model = build_model(X_train, y_train, params)
        logger.info("Model building process completed successfully.")
        
        # Save the trained model to a file
        model_dir = './models'
        save_model(model, model_dir)
        logger.info(f"Model trained finished and saved successfully at {model_dir}")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
    
if __name__ == "__main__":
    main()
