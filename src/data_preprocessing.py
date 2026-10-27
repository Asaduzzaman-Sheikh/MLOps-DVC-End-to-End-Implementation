# Import necessary libraries
import os 
import pandas as pd
import logging

from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources silently
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ----------------------------------------------------------- LOGGING SETUP -----------------------------------------------------------
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ----------------------------------------------------------- DATA PREPROCESSING STAGE -----------------------------------------------------------

def preprocess_data(train_data: pd.DataFrame, test_data: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        # Initialize the LabelEncoder
        label_encoder = LabelEncoder()
        
        # Encode categorical columns in the training data
        train_data[target_column] = label_encoder.fit_transform(train_data[target_column])    
        logger.debug("Target column in training data encoded successfully.")
            
        # Encode categorical columns in the testing data using the same encoders
        test_data[target_column] = label_encoder.transform(test_data[target_column])    
        logger.debug("Target column in testing data encoded successfully.") 
        
        
        logger.debug("Data preprocessing completed successfully.")
        logger.info("Successfully encoded the target column in both training and testing data.")    
        
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise  
    
# Initialize the PorterStemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def text_transformation(text: str) -> str:
    try: 
        if not isinstance(text, str): # Safety check for NaN values
            return ""
            
        text = text.lower()
        text = nltk.word_tokenize(text)

        # 1. Remove special characters and numbers
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)

        # 2. Remove stop words and punctuation
        text = y[:]
        y.clear()

        for i in text:
            if i not in stop_words and i not in string.punctuation:
                y.append(i)

        # 3. Stemming
        text = y[:] # Save the cleaned words
        y.clear()   

        for i in text:
            y.append(ps.stem(i))

        # Join the stemmed words back into a string
        return " ".join(y)
    
    except Exception as e:
        logger.error(f"Error during text transformation: {e}")
        raise
        
# ------------------------------------------------------------ MAIN FUNCTION -----------------------------------------------------------
def main():
    try:
        # Extract the training and testing data from the CSV files
        train_data = pd.read_csv('./data/train/train_data.csv')
        test_data = pd.read_csv('./data/test/test_data.csv')
        logger.debug("Training and testing data loaded successfully.")
        
        # Preprocess the data (Encode Target)
        target_column = 'Target'
        train_data, test_data = preprocess_data(train_data, test_data, target_column)
        
        # Transform the text data in the training and testing datasets
        logger.info("Starting text transformation (this may take a minute)...")
        train_data['Feature'] = train_data['Feature'].apply(text_transformation)
        test_data['Feature'] = test_data['Feature'].apply(text_transformation)
        logger.info("Text transformation completed successfully for all rows.")
        
        # Setup directories for transformed data
        interphase_dir = './data/interphase'
        os.makedirs(interphase_dir, exist_ok=True)
        
        train_dir = os.path.join(interphase_dir, 'train_transformed')
        test_dir = os.path.join(interphase_dir, 'test_transformed')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        train_transformed_path = os.path.join(train_dir, 'train_transformed.csv')
        test_transformed_path = os.path.join(test_dir, 'test_transformed.csv')
        
        # Save the preprocessed and transformed training and testing data to CSV files
        train_data.to_csv(train_transformed_path, index=False)
        test_data.to_csv(test_transformed_path, index=False)
        
        logger.info(f"Transformed train data saved to {train_transformed_path}")
        logger.info(f"Transformed test data saved to {test_transformed_path}")
        logger.info("Data preprocessing and transformation pipeline completed successfully.")
     
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {fnf_error}")
        raise
    
    except pd.errors.EmptyDataError as empty_error:
        logger.error(f"Empty data error: {empty_error}")
        raise
       
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()