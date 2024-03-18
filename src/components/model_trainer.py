import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from src.exception import Error
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_dir: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model_trainer_config = config  # Assigning the config attribute

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                'LogisticRegression': LogisticRegression(max_iter=500),
                'RandomForestClassifier': RandomForestClassifier(),
                'GradientBoostingClassifier': GradientBoostingClassifier(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'KNeighborsClassifier': KNeighborsClassifier(),
                'GaussianNB': GaussianNB(),
                'SVC': SVC(),
                'XGBClassifier': XGBClassifier(),
                'CatBoostClassifier': CatBoostClassifier(),
            }
            param = {
                'LogisticRegression': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
                'RandomForestClassifier': {
                    'n_estimators': [10, 100],
                    'max_features': ['sqrt', 'log2'],
                },
                'GradientBoostingClassifier': {
                    'n_estimators': [50, 100, 500],
                    'learning_rate': [0.001, 0.01, 0.1],
                },
                'DecisionTreeClassifier': {'criterion': ['gini', 'entropy']},
                'KNeighborsClassifier': {'n_neighbors': [3, 5, 11, 19]},
                'GaussianNB': {},
                'SVC': {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]},
                'XGBClassifier': {
                    'n_estimators': [100, 500],
                    'learning_rate': [0.001, 0.01, 0.1],
                },
                'CatBoostClassifier': {
                    'iterations': [100, 500],
                    'learning_rate': [0.001, 0.01, 0.1],
                },
            }
            self.logger.info(f'Evaluating models')
            model_reports = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=param)

            # To get best model score and name from dict
            best_model_name, best_model_report = max(model_reports.items(), key=lambda item: item[1]['test']['accuracy'])

            best_model = models[best_model_name]

            if best_model_report['test']['accuracy'] < 0.7:
                raise Exception("No best model found")

            logging.info("Best found model on both training and testing dataset")

            save_object(
                filepath=self.config.trained_model_dir,  # Corrected argument name
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)

            return accuracy

        except Exception as e:
            self.logger.error(f'Error training model: {e}')
            raise Error(f'Error training model: {e}', error_detail=str(e))
