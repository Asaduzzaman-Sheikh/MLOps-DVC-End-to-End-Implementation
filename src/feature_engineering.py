# Import necessary libraries
import os
import pandas as pd
import logging

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ------------------------------------------------------ LOGGING SETUP ------------------------------------------------------ #

log_dir = 'logs'
os.makedirs(log_dir, exist_ok = True)

# Create a logger object
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

# Create a console handler for the logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a file handler for the logger
file_path =  os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)

# Create a formatter for the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Apply the formatter to the handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ------------------------------------------------------ FEATURE ENGINEERING STAGE ------------------------------------------------------ #
# Define the function for loading data and performing feature engineering
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and perform feature engineering.
    
    Args:
        file_path (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: The loaded and processed data.
    """
    try:
        # Load the data into a pandas DataFrame
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
    
# Define the function for feature engineering
def feature_engineering(train_data: pd.DataFrame, test_data: pd.DataFrame,text_cols, target_col, max_features: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform feature engineering on the training and testing data.
    
    Args:
        train_data (pd.DataFrame): The training data as a pandas DataFrame.
        test_data (pd.DataFrame): The testing data as a pandas DataFrame.
        max_features (int): The maximum number of features to consider for the TF-IDF vectorizer.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The transformed training and testing data.
    """
    try:
        # Initialize the TF-IDF Vectoizer
        tfidf_vectorizer = TfidfVectorizer(max_features = max_features)
        
        # Seperate the training and testing data into features and target
        X_train_text = (
            train_data[text_cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)      
        )
        X_test_text = (
            test_data[text_cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )
        y_train = train_data[target_col].values
        y_test  = test_data[target_col].values
        
        # Vectorize the text data using TF-IDF
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
        X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
        
        train_data_transformed = pd.DataFrame(X_train_tfidf.toarray(), columns = tfidf_vectorizer.get_feature_names_out())
        train_data_transformed['Target'] = y_train
        
        
        test_data_transformed = pd.DataFrame(X_test_tfidf.toarray(), columns = tfidf_vectorizer.get_feature_names_out())
        test_data_transformed['Target'] = y_test
        
        logger.debug("Feature engineering completed successfully.")
        logger.debug(f"Transformed training data shape: {train_data_transformed.shape}")
        logger.debug(f"Transformed testing data shape: {test_data_transformed.shape}")
        return train_data_transformed, test_data_transformed
    except ValueError as ve:
        logger.error(f"Value error during feature engineering: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise
    
# Define the function for saving the transformed data
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, base_dir: str) -> None:
    """
    Save the transformed training and testing data to CSV files in the specified directory.
    """
    try:
        # Create the base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok = True)
        
        file_path_train = os.path.join(base_dir, 'train_transformed.csv')
        file_path_test = os.path.join(base_dir, 'test_transformed.csv')
        
        train_data.to_csv(file_path_train, index = False)
        test_data.to_csv(file_path_test, index = False)
        logger.debug(f"Transformed training data saved to {file_path_train}")
        logger.debug(f"Transformed testing data saved to {file_path_test}")
    except Exception as e:
        logger.error(f"Error saving transformed data: {e}")
        raise


# ------------------------------------------------------ MAIN FUNCTION ------------------------------------------------------ #
def main():
    try:
        # Load the data 
        train_data = load_data('./data/interphase/train_transformed/train_transformed.csv')
        test_data = load_data('./data/interphase/test_transformed/test_transformed.csv')
        logger.debug("Data loaded successfully for feature engineering.")
        
        # Perform feature engineering
        max_features = 50
        target_col = 'Target'
        text_cols = ['Feature']  # Specify the text columns to be used for feature engineering
        train_data_transformed, test_data_transformed = feature_engineering(train_data, test_data, text_cols, target_col, max_features)
        
        # Save the transformed data to the specified directory
        base_dir = './data/processed'
        save_data(train_data_transformed, test_data_transformed, base_dir)
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
    
if __name__ == "__main__":
    main()
