import os
import sys
from dataclasses import dataclass
from typing import List
import pickle

# Add root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import Error
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)  # Instantiate the logger

    def transform_data(self):
        try:
            numerical_columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            categorical_columns = []

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
                ])

            logging.info(f'Creating preprocessor')
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', numeric_transformer, numerical_columns),
                    ('cat', categorical_transformer, categorical_columns)
                    ])

            return preprocessor

        except Exception as e:
            self.logger.error(f'Error transforming data: {e}')
            raise Error(f'Error transforming data: {e}')

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.transform_data()

            target_column_name = "Outcome"
            numerical_columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                                "DiabetesPedigreeFunction", "Age"]
            categorical_columns = []

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(self.config.preprocessor_obj_file_path, preprocessing_obj)

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path,
            )
        except Exception as e:
            self.logger.error(f'Error transforming data: {e}')
            raise Error(f'Error transforming data: {e}')
