import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_predictions(predictions, output_path):
    """
    Save predictions to a CSV file.
    """
    pd.DataFrame(predictions, columns=['predicted_sales']).to_csv(output_path, index=False)