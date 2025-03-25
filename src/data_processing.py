import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def clean_data(data):
    """
    Clean the data by handling missing values and removing duplicates.
    """
    data = data.dropna()  # Drop rows with missing values
    data = data.drop_duplicates()  # Remove duplicate rows
    return data

def preprocess_data(data):
    """
    Preprocess the data for model training (e.g., feature engineering, encoding).
    """
    # Example: Convert date to datetime and extract features
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
    return data

def save_cleaned_data(data, output_path):
    """
    Save the cleaned and preprocessed data to a CSV file.
    """
    data.to_csv(output_path, index=False)