import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str) -> dict:
    """Load parameters from yaml"""
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not Found: %s', params_path)
    except yaml.YAMLError as e:
        logger.error('YAML error: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected Error: %s',e)
        raise

def load_data(data_path: str) -> pd.DataFrame:
    """Load file from folder artifacts"""
    try:
        df = pd.read_csv(data_path)
        logger.debug('Data load from : %s', data_path)
        return df
    except Exception as e:
        logger.error('Failed to load csv file : %s',e)
        raise

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    """Preprocessing the data by handling missing values, duplicated, and empty string"""
    try:
        #Remove missing values
        df.dropna(inplace=True)

        #Remove duplicates
        df.drop_duplicates(inplace=True)

        #Remove rows with empty string
        df = df[df['clean_comment'].str.strip() != '']

        logger.debug('Data preprocessing completed: Missing values, duplicates, and empty string')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s',e)
        raise

def save_data(train_data: pd.DataFrame, test_data:pd.DataFrame, data_path:str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist"""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)

        logger.debug('Train and test data saved to %s',raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s',e)
        raise

def main():
    try:
        # load parameters from the params.yaml in the root directory
        params      = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size   = params['data_ingestion']['test_size']
        random_state= params['data_ingestion']['random_state']

        df = load_data(data_path='artifacts/Reddit_Data.csv')

        final_df = preprocess_data(df)

        train_df, test_df = train_test_split(final_df, test_size=test_size, random_state=random_state)

        save_data(train_df, test_df, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../data'))
    
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s',e)
        raise(f"Error: {e}")

if __name__ == "__main__":
    main()

