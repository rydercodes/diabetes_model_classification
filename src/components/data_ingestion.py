import os
import sys
import logging
import pandas as pd
# Add root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.exception import Error
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from src.utils import save_object

@dataclass
class DataIngestionConfig:
    data_path: str = os.path.join('data', 'diabetes.csv')
    test_size: float = 0.2
    random_state: int = 42

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        try:
            self.logger.info(f'Loading data from {self.config.data_path}')
            data = pd.read_csv(self.config.data_path)
            return data
        except Exception as e:
            raise Error(f'Error loading data: {e}')

    def split_data(self, data: pd.DataFrame):
        try:
            script_dir = os.path.abspath(sys.path[0])
            root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
            artifacts = os.path.join(root_dir, 'artifacts')
            os.makedirs(artifacts, exist_ok=True)

            data.to_csv(os.path.join(artifacts, 'raw.csv'), index=False, header=True)
            self.logger.info(f'Splitting data into train and test sets')
            train, test = train_test_split(data, test_size=self.config.test_size, random_state=self.config.random_state)

            train.to_csv(os.path.join(artifacts, 'train.csv'), index=False, header=True)
            test.to_csv(os.path.join(artifacts, 'test.csv'), index=False, header=True)
            return train, test
        except Exception as e:
            self.logger.error(f'Error splitting data: {e}')
            raise Error(f'Error splitting data: {e}', error_detail=str(e))

if __name__ == '__main__':
    config = DataIngestionConfig()
    di = DataIngestion(config)
    data = di.load_data()
    train, test = di.split_data(data)

    transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation(transformation_config)

    train_path = os.path.join('artifacts', 'train.csv')
    test_path = os.path.join('artifacts', 'test.csv')
    data_transformation.initiate_data_transformation(train_path=train_path, test_path=test_path)

    model_trainer_config = ModelTrainerConfig()
    model_trainer = ModelTrainer(model_trainer_config)

    report = model_trainer.train_model(X_train=train.iloc[:, :-1], y_train=train.iloc[:, -1],
                                       X_test=test.iloc[:, :-1], y_test=test.iloc[:, -1])

    best_model_name = max(report, key=lambda x: report[x]['test']['accuracy'])
    print(f"Best model: {best_model_name}")
    # Assuming report is a dictionary containing evaluation metrics
    report_df = pd.DataFrame(report).T

    # Print the DataFrame
    print("Evaluation Metrics:")
    print(report_df)
