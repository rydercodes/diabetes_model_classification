import os
import sys
import pandas as pd
from src.utils import load_object
from src.exception import Error
# Add root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.abspath(os.path.join("artifacts", "model.pkl"))
            preprocess_path = os.path.abspath(os.path.join("artifacts", "preprocessor.pkl"))
            model = load_object(filepath=model_path)
            preprocess = load_object(filepath=preprocess_path)
            features = preprocess.transform(features)
            prediction = model.predict(features)
            return prediction
        except Exception as e:
            raise Error(f"Error in predict pipeline: {e}")

class CustomData:
    def __init__(
            self,
            Pregnancies: int,
            Glucose: int,
            BloodPressure: int,
            SkinThickness: int,
            Insulin: int,
            BMI: float,
            DiabetesPedigreeFunction: float,
            Age: int
            ):
        self.Pregnancies = Pregnancies
        self.Glucose = Glucose
        self.BloodPressure = BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin = Insulin
        self.BMI = BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self.Age = Age

    def to_dict(self):
        return {
            "Pregnancies": self.Pregnancies,
            "Glucose": self.Glucose,
            "BloodPressure": self.BloodPressure,
            "SkinThickness": self.SkinThickness,
            "Insulin": self.Insulin,
            "BMI": self.BMI,
            "DiabetesPedigreeFunction": self.DiabetesPedigreeFunction,
            "Age": self.Age
        }
