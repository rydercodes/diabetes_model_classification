import os
import sys
from flask import Flask, request, render_template
import pandas as pd

# Add root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pipeline.predict_pipeline import PredictPipeline

application = Flask(__name__)
app = application

# Define the CustomData class
class CustomData:
    def __init__(self, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
        self.Pregnancies = Pregnancies
        self.Glucose = Glucose
        self.BloodPressure = BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin = Insulin
        self.BMI = BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self.Age = Age
    
    def get_data_as_data_frame(self):
        # Convert input data to a DataFrame
        data = {
            'Pregnancies': [self.Pregnancies],
            'Glucose': [self.Glucose],
            'BloodPressure': [self.BloodPressure],
            'SkinThickness': [self.SkinThickness],
            'Insulin': [self.Insulin],
            'BMI': [self.BMI],
            'DiabetesPedigreeFunction': [self.DiabetesPedigreeFunction],
            'Age': [self.Age]
        }
        return pd.DataFrame(data)

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data = CustomData(
            Pregnancies=request.form.get('Pregnancies'),
            Glucose=request.form.get('Glucose'),
            BloodPressure=request.form.get('BloodPressure'),
            SkinThickness=request.form.get('SkinThickness'),
            Insulin=request.form.get('Insulin'),
            BMI=float(request.form.get('BMI')),
            DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction')),
            Age=float(request.form.get('Age'))
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        if results[0] == 0:
            prediction_text = "The sample measurements do not indicate diabetes."
        elif results[0] == 1:
            prediction_text = "The sample measurements show signs of diabetes."
        else:
            prediction_text = "Prediction result is unknown."

        return render_template('index.html', results=results[0], prediction_text=prediction_text)

if __name__=="__main__":
    app.run(host="0.0.0.0")
