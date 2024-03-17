import os
import pickle
from sklearn.metrics import accuracy_score, r2_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from src.exception import Error


def save_object(filepath: str, obj) -> None:
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'wb') as fileObject:
            pickle.dump(obj, fileObject)
    except Exception as e:
        raise Error(f'Error saving object: {e}')


def load_object(filepath: str):
    try:
        with open(filepath, 'rb') as fileObject:
            obj = pickle.load(fileObject)
        return obj
    except Exception as e:
        raise Error(f'Error loading object: {e}')


def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        reports = {}

        for model_name, model in models.items():
            params = param[model_name]
            grid = GridSearchCV(model, params, cv=3)
            grid.fit(X_train, y_train)

            model.set_params(**grid.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score_r2 = r2_score(y_train, y_train_pred)
            test_model_score_r2 = r2_score(y_test, y_test_pred)

            train_model_score_accuracy = accuracy_score(y_train, y_train_pred)
            test_model_score_accuracy = accuracy_score(y_test, y_test_pred)

            train_model_score_recall = recall_score(y_train, y_train_pred)
            test_model_score_recall = recall_score(y_test, y_test_pred)

            train_model_score_precision = precision_score(y_train, y_train_pred)
            test_model_score_precision = precision_score(y_test, y_test_pred)

            report = {
                'train': {
                    'r2': train_model_score_r2,
                    'accuracy': train_model_score_accuracy,
                    'recall': train_model_score_recall,
                    'precision': train_model_score_precision
                },
                'test': {
                    'r2': test_model_score_r2,
                    'accuracy': test_model_score_accuracy,
                    'recall': test_model_score_recall,
                    'precision': test_model_score_precision
                }
            }

            reports[model_name] = report

        return reports
    except Exception as e:
        raise Error(f'Error evaluating model: {e}')
