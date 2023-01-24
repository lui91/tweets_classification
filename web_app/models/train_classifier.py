import pandas as pd
from sqlalchemy import create_engine
import nltk
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import sys
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import os


def load_data(database_filepath) -> pd.DataFrame:
    """ Read data from SQL lite database """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_Fact", engine)
    X = df["message"]
    Y = df[df.columns[4:]]
    return X, Y


def tokenize(text) -> list:
    """ Pre process a text line"""
    lower_case = text.lower()
    words = re.sub(r'[^a-zA-Z0-9]', " ", lower_case)
    words = words.split()
    return words


def build_model() -> GridSearchCV:
    """ Build process pipeline and perform grid 
        search for hyperparameter tuning """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('predictor', MultiOutputClassifier(RandomForestClassifier()))
    ])
    grid_params = {
    "predictor__estimator__criterion": ["gini", "entropy"],
    "predictor__estimator__max_features": ["sqrt", "log2"]
    }
    cv = GridSearchCV(pipeline, grid_params)
    return cv


def evaluate_model(model, X_test, Y_test) -> None:
    """ Calculate precision, recall and F1 score """
    weighted_recall = []
    weighted_precision = []
    weighted_f1 = []

    Y_pred = model.predict(X_test)
    for i, key in enumerate(Y_test):
        report = classification_report(Y_pred[:, i], Y_test.iloc[:, i], output_dict=True)
        weighted_precision.append(report['weighted avg']['precision'])
        weighted_recall.append(report['weighted avg']['recall'])
        weighted_f1.append(report['weighted avg']['f1-score'])
    print(f"Precision: {np.mean(np.asarray(weighted_precision))}")
    print(f"Recall: {np.mean(np.asarray(weighted_recall))}")
    print(f"F1 score: {np.mean(np.asarray(weighted_f1))}")


def save_model(model, model_filepath: str):
    """ Save the best configuration of the model found with grid search """
    file_name = Path(model_filepath)
    if file_name.exists():
        os.remove(file_name.as_posix())
    estimator = model.best_estimator_
    dump(estimator, model_filepath)


def main():
    """
    Automated data ingestion, cleaning, Multi output model training 
    using Random forest classifiet, and hyperparameter tuning.
    Input: Database path, output model name.
    Output: Trained model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()