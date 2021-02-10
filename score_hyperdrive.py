import json
import numpy as np
import os
from sklearn.externals import joblib
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import QuantileTransformer

def init():
    # load the model
    global model, scaler
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'outputs')
    model = joblib.load(os.path.join(model_path, 'model.joblib'))
    scaler = joblib.load(os.path.join(model_path, 'qt.joblib'))

def run(input_data):
    try:
        data = np.array(json.loads(input_data)['data'])
        result = model.predict(scaler.transform(data))
        result = result.round(decimals=2)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error