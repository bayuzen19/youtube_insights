import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path:str) -> dict:
    '''Load parameters from a yaml file'''
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrived from %s', params_path)
        return params

    except FileNotFoundError:
        logger.error('File not found : %s', params_path)
        raise

    except yaml.YAMLError as e:
        logger.error('YAML error : %s', e)
        raise
    
    except Exception as e:
        logger.error('Unexpected error: %s',e)
        raise

def load_data(file_path:str) -> pd.DataFrame:
    '''Load data from csv'''
    try:
        df = pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug('Data loaded and NaNs filled from %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s',e)
        raise

def apply_bow(train_data:pd.DataFrame, max_features:int, ngram_range:Tuple) -> Tuple:
    '''Apply BoW with ngrams to dataset'''
    try:
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        X_train_vec = vectorizer.fit_transform(X_train)
        logger.debug(f"BoW transformation complete. Train shape: {X_train_vec.shape}")

        with open(os.path.join(get_root_directory(), 'bow_vectorizer.pkl'),'wb') as f:
            pickle.dump(vectorizer,f)
        
        logger.debug('BoW applied unigrams and data transformed')
        return X_train_vec, y_train
    except Exception as e:
        logger.error('Error during BoW transformation: %s',e)
        raise

def train_logreg(X_train: np.array, y_train:np.array, C:float,max_iter:int, solver:str, random_state:int) -> LogisticRegression:
    '''
    Train Logistic Regresison
    params :
      - X_train after vectorization
      - y_train
      - C : strength regulizations
      - solver : Coef optimization
      - random_state : randomization
    
    '''
    try:
        best_model = LogisticRegression(
            random_state=random_state,
            C=C,
            max_iter=max_iter,
            solver=solver
        )

        best_model.fit(X_train,y_train)
        logger.debug('Logistic Regression model training completed')
        return best_model
    except Exception as e:
        logger.info('Error during training logistic regression: %s',e)
        raise

def save_model(model, file_path:str) -> None:
    '''Save model training'''
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s',file_path)
    except Exception as e:
        logger.error('Error occured while saving the model: %s',e)
        raise

def get_root_directory() -> str:
    '''Get the root directory'''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        # get root directory and resolve the path from params.yaml
        root_dir = get_root_directory()

        # load parameter from root directory
        params      = load_params(os.path.join(root_dir,'params.yaml'))
        max_features= params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])

        #load model parameter
        C        = params['model_building']['C']
        max_iter = params['model_building']['max_iter']
        solver   = params['model_building']['solver']
        random_state = params['data_ingestion']['random_state']

        #training data
        train_data = load_data(os.path.join(root_dir,'data/interim/train_processed.csv'))

        #apply BoW transformation
        X_train_vec, y_train = apply_bow(train_data, max_features, ngram_range)

        # Training Model
        best_model = train_logreg(X_train_vec, y_train, C=C,max_iter=max_iter,solver=solver,random_state=random_state)

        save_model(best_model, os.path.join(root_dir,'logistic_regression.pkl'))
    except Exception as e:
        logger.error('Failed to complete the feature engineering and model build process: %s',e)
        print(f"Error: {e}")
    
if __name__=="__main__":
    main()