import sys
import os
from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Create the 'outputs' directory if it doesn't exist
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Load the trained model
model_path = 'outputs/model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

model = joblib.load(model_path)

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    """Home endpoint to provide instructions."""
    return """
    Welcome to the Demand Prediction API!<br><br>
    Use POST /predict with JSON data to get predictions.<br>
    Example JSON payload:<br>
    <pre>
    [
        {"date": "2023-10-01", "product_id": "A"},
        {"date": "2023-10-02", "product_id": "B"}
    ]
    </pre>
    """

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to make demand predictions."""
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)
        
        # Check if the input is a list
        if not isinstance(data, list):
            return jsonify({"error": "Input must be a list of objects."}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure required columns are present
        required_columns = ['date', 'product_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing required columns: {missing_columns}"}), 400
        
        # Preprocess the data
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df = df.drop(columns=['date'])
        df = pd.get_dummies(df, columns=['product_id'], drop_first=True)
        
        # Ensure the input data has the same columns as the training data
        # Add missing columns with default values (0)
        expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        if expected_columns is not None:
            for column in expected_columns:
                if column not in df.columns:
                    df[column] = 0
            df = df[expected_columns]  # Reorder columns to match training data
        
        # Make predictions
        predictions = model.predict(df)
        
        # Return predictions as JSON
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)