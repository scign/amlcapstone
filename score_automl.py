import json
import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib

def init():
    # load the model
    global model
    model_path = os.getenv('AZUREML_MODEL_DIR')
    model = joblib.load(os.path.join(model_path, 'model.pkl'))

def run(input_data):
    try:
        data = pd.DataFrame(json.loads(input_data)['data'], columns=[
            'fixed acidity', 'volatile acidity', 'citric acid',
            'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])
        result = model.predict(data)
        result = result.round(decimals=2)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error