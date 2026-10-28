# Import the necessary libraries
from ast import main
import os
import pandas as pd
import logging
import yaml

from sklearn.model_selection import train_test_split
# ----------------------------------------------------------- LOGGING SETUP -----------------------------------------------------------
# Make the directory for the log file if it doesn't exist
# Define the path, then create the directory
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

file_path = os.path.join(log_dir, 'data_ingestion.log')

# Set up logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

# Set up a console handler for the logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Set up a file handler for the logger
file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)

# Set up a formatter for the log messages and apply it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Function to load the parameters
def load_params(file_path: str) -> dict:
    """ 
        This function will take a 'file_path' to load all the parameters 

    """
    
    try:
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.debug(f"Parameter is retrieved from {file_path}")
            
            return params
    except FileNotFoundError:
        logger.error("File not found at {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: {e}") 
        raise
    except Exception as e:
        logger.error("An unexpected error occurs: {e}")
        raise

# ----------------------------------------------------------- DATA INGESTION STAGE -----------------------------------------------------------

# Define the load data function
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        # Load the data into a pandas DataFrame
        data = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}.")
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
    
    
# Define the process data function
def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process the data by performing necessary cleaning and transformations.

    Args:
        data (pd.DataFrame): The data to be processed.

    Returns:
        pd.DataFrame: The processed data as a pandas DataFrame.
    """
    try:
        processed_data = data.copy()
        processed_data.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        processed_data.columns = ['Target', 'Feature']
        processed_data.drop_duplicates(inplace = True)
        logger.debug("Drop duplicates and unnecessay columns successfully")
        logger.debug("Data processed successfully.")
        
        return processed_data
    except KeyError as e:
        logger.error(f"Key error during the data processing: {e}. Please check the column names.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during data processing: {e}")
        raise
    
# Define the save data function
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, base_data_path: str) -> None:
    
    """
    Save the processed data to a CSV file.

    Args:
        train_data (pd.DataFrame): The training data.
        test_data (pd.DataFrame): The test data.
        base_data_path (str): The base path to the data directory.
    """
    try:
        train_dir = os.path.join(base_data_path, 'train')
        test_dir = os.path.join(base_data_path, 'test')
        
        # Create directories if they don't exist
        os.makedirs(train_dir, exist_ok = True)
        os.makedirs(test_dir, exist_ok = True)
        
        # Define the file paths for train and test data
        train_file_path = os.path.join(train_dir, 'train_data.csv')
        test_file_path =  os.path.join(test_dir, 'test_data.csv')
        
        # Save the data to CSV files
        train_data.to_csv(train_file_path, index = False)
        test_data.to_csv(test_file_path, index = False)
        logger.debug(f"Train and Test data saved successfully to {train_file_path} and {test_file_path}.")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving data: {e}")
        raise
    
    # ----------------------------------------------------------- MAIN FUNCTION -----------------------------------------------------------
def main():
    try:
        base_data_path = './data'
        raw_data_url = 'https://raw.githubusercontent.com/vikashishere/YT-MLOPS-Complete-ML-Pipeline/refs/heads/main/experiments/spam.csv'
        
        # Load raw data
        raw_data = load_data(raw_data_url)
        logger.debug("Raw data loaded successfully")
        
        # Save raw data
        raw_dir = os.path.join(base_data_path, 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        raw_file_path = os.path.join(raw_dir, 'raw_spam.csv')
        raw_data.to_csv(raw_file_path, index=False)
        logger.debug(f"Raw spam data saved successfully to {raw_file_path}.")    
        
        # Process data
        processed_data = process_data(raw_data)
        logger.debug("Data processed successfully")
        
        # Split data
        
        # Load the parameters
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion']['random_state']
        train_data, test_data = train_test_split(
            processed_data,
            test_size=test_size,
            random_state=random_state,
            stratify=processed_data['Target']
        )
        logger.debug(f"Data split into train and test sets with test size = {test_size} and random state = {random_state}.")
        
        # Save processed data
        save_data(train_data, test_data, base_data_path)
        logger.debug(f"Processed train and test data saved successfully to {base_data_path}.")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main function: {e}")
        raise
    
    
    
if __name__ == "__main__":
    main()
            
    

    
    