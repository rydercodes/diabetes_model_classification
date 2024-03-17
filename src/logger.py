import logging
import os
from datetime import datetime


LOG_FILE = os.path.join(os.path.dirname(__file__), 'logs', 'log.txt')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


# Ensure the directory structure exists
log_directory = os.path.dirname(LOG_FILE)
os.makedirs(log_directory, exist_ok=True)


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT
    )