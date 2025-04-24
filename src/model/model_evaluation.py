import numpy as np
import pandas as pd
import os
import pickle
import logging
import yaml

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import json

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str) -> pd.DataFrame:
    '''Load csv data'''
    try:
        df = pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug('Data loaded and NaNs filled from %s',file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise

def load_model(model_path:str):
    '''load model'''
    try:
        with open(model_path,'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s',model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s',model_path,e)
        raise

def load_vectorizer(vectorizer_path:str) -> CountVectorizer:
    '''Load the BoW vectorizer'''
    try:
        with open(vectorizer_path,'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('BoW vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error(f'Error loading vectorizer from %s: %s', vectorizer_path,e)
        raise

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

def evaluate_model(model, X_test:np.ndarray, y_test:np.ndarray):
    '''Evaluate the model and log classification metrics and confusion matrix'''
    try:
        #predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test,y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        logger.debug('Model evaluation completed')

        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def log_confusion_matrix(cm, dataset_name):
    '''Log confussion matrix as an artifact'''
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()

def save_model_info(run_id:str, model_path:str, file_path:str) -> None:
    '''save model run id and path to a json file'''
    try:
        #create a dictionary with the info want to save
        model_info = {
            'run_id':run_id,
            'model_path':model_path
        }

        #save the dictionary as json file
        with open(file_path,'w') as file:
            json.dump(model_info,file,indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occured while saving the modol iunfo: %s',e)
        raise

def main():
    mlflow.set_tracking_uri(uri="http://localhost:8080")
    mlflow.set_experiment('youtube-insight-pipeline-runs')

    with mlflow.start_run(nested=True) as run:
        try:
            #load parameters from yaml file
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../'))
            params   = load_params(os.path.join(root_dir, 'params.yaml'))

            #load parameters
            for key, value in params.items():
                mlflow.log_param(key,value)
            
            model = load_model(os.path.join(root_dir, 'logistic_regression.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'bow_vectorizer.pkl'))

            #load test data for signature inference
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            #prepare test data
            X_test_vec = vectorizer.transform(test_data['clean_comment'].values)
            y_test     = test_data['category'].values

            # create dataframe for signature inference
            input_example = pd.DataFrame(X_test_vec.toarray()[:10], columns=vectorizer.get_feature_names_out())

            # Infer the signature
            signature = infer_signature(input_example, model.predict(X_test_vec[:10]))

            #log model with signature
            mlflow.sklearn.log_model(
                model,
                'logistic_regression',
                signature=signature,
                input_example=input_example
            )

            #save model info
            artifact_uri = mlflow.get_artifact_uri()
            model_path   = f'{artifact_uri}/logistic_regression'
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')

            #log the vectorizer as an artifact
            mlflow.log_artifact(os.path.join(root_dir, 'bow_vectorizer.pkl'))

            # evaluate model and get metrics
            report, cm = evaluate_model(model, X_test_vec, y_test)

            #log classification report metrics for the test data
            for label, metrics in report.items():
                if isinstance(metrics,dict):
                    mlflow.log_metrics({
                        f'test_{label}_precision':metrics['precision'],
                        f'test_{label}_recall':metrics['recall'],
                        f'test_{label}_f1-score':metrics['f1-score']
                    })

            # log confusion matrix
            log_confusion_matrix(cm, 'test data')

            #set tag
            mlflow.set_tag('model_type','logistic_regression')
            mlflow.set_tag('task','Sentiment Analysis')
            mlflow.set_tag('dataset','YouTube Comments')
        
        except Exception as e:
            logger.error('Failed to complete model evaluation: {e}')
            print(f'Error: {e}')

if __name__=="__main__":
    main()