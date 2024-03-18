# Diabetes Prediction Model

This repository contains code for a machine learning model that predicts the likelihood of a person having diabetes based on various health metrics. The model is trained on a dataset containing information about patients, including factors such as `glucose level`, `blood pressure`, `BMI`, etc.

## Dataset

The dataset used for training the model is stored in the `data` directory. It is a CSV file named `diabetes.csv` containing information about patients.

## Setup

To set up the environment for running the model, follow these steps:

1. Clone the repository: `git clone https://github.com/rydercodes/diabetes_model_classification.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the `data_ingestion.py` script to preprocess the data: `python src/components/data_ingestion.py`

## Usage

Once the model is trained, you can use it to make predictions on new data. You can either load the trained model directly or use the provided scripts for data ingestion and model training. 

## Model Deployment

The trained model can be deployed locally using Flask to create a REST API for serving predictions. To deploy the model and make predictions on new data, run the `predict.py` script:
`python predict.py`


![](main.png)

## Contributors

- Jaber Rahimifard <Jaber.rahimifard@outlook.com>
