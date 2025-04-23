import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#nltk download
nltk.download('wordnet')
nltk.download('stopwords')

#======= Preprocessing ============
def preprocess_comment(comment):
    '''Apply preprocessing transformation to a comment.'''
    try:
        logger.info('Starting preprocessing word')
        #convert to lowercase
        comment = comment.lower()

        #remove trailing and leading whitespace
        comment = comment.strip()

        #remove new line
        comment = re.sub(r'\n',' ',comment)

        #remove non-alphanumeric characters
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '',comment)

        #remove stopwrods
        stop_words = set(stopwords.words('english')) - {'not','but','however','no','yet'}
        comment    = ' '.join([word for word in comment.split() if word not in stop_words])

        #lemmatize
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        logger.info('successfully preprocessing word')

        return comment
    
    except Exception as e:
        logger.error(f'Error in preprocessing comment: {e}')
        raise

def normalize_text(df):
    '''Apply preprocessing to the text data in the dataframe'''
    try:
        logger.info('Starting normalize text')
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)

        logger.info('Successfully normalize text')
        return df
    
    except Exception as e:
        logger.error(f'Error during text normalization: ',{e})
        raise

def save_data(train_data, test_data, data_path):
    '''Save the process train and test datasets'''
    try:
        logger.info('Starting to save train and test datasets')
        interim_data_path = os.path.join(data_path,'interim')
        logger.debug(f'Creating directory')

        os.makedirs(interim_data_path,exist_ok=True)
        logger.debug(f'Directory {interim_data_path} create or already created')

        train_data.to_csv(os.path.join(interim_data_path, 'train_processed.csv'),index=False)
        test_data.to_csv(os.path.join(interim_data_path, 'test_processed.csv'),index=False)

        logger.info(f'Successfully to save train and test to {interim_data_path}')

    except Exception as e:
        logger.error(f'Error occurred while saving data: {e}')
        raise

def main():
    try:
        logger.debug('Starting data preprocessing...')

        #Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data  = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded successfully')

        #preprocessing
        train_processed_data = normalize_text(train_data)
        test_processed_data  = normalize_text(test_data)

        #save the processed data
        save_data(train_processed_data, test_processed_data, data_path='./data')
        logger.debug('successfully process')
    
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s',e)
        print(f"Error: {e}")

if __name__=='__main__':
    main()