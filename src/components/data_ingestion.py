import os 
import sys
import logging  # Add this import statement for logging

# Add root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.exception import Error
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    data_path: str=os.path.join('data',"diabetes.csv")
    test_size: float=0.2
    random_state: int=42
    

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)  # Instantiate the logger

    def load_data(self):
        try:
            self.logger.info(f'Loading data from {self.config.data_path}')
            data = pd.read_csv(self.config.data_path)
            return data
        except Exception as e:
            raise Error(f'Error loading data: {e}')  # Removed sys from here

    def split_data(self, data: pd.DataFrame):
        try:
            # Get the directory of the script being run
            script_dir = os.path.abspath(sys.path[0])

            # Navigate up from the script's directory to the root directory
            root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

            # Create the 'artifacts' directory in the root directory
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
            raise Error(f'Error splitting data: {e}')

if __name__ == '__main__':
    config = DataIngestionConfig()
    di = DataIngestion(config)
    data = di.load_data()
    train, test = di.split_data(data)
    print(train.head())
    print(test.head())
    print(f'Train shape: {train.shape}')
    print(f'Test shape: {test.shape}')
