import joblib
import pandas as pd

def load_model(model_path):
    """
    Load a trained model from a file.
    """
    return joblib.load(model_path)

def predict_demand(model, new_data):
    """
    Make predictions using the trained model.
    """
    return model.predict(new_data)