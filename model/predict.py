import pandas as pd
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# which will be automatic later using MLflow 
MODEL_VERSION = '1.0.0'

def predict_output(user_input: dict):
    
    input_df = pd.DataFrame([user_input])

    output=  model.predict(input_df)[0] # retriving first value which we only need 

    return output
