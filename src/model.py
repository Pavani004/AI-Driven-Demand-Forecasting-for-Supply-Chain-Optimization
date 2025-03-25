from xgboost import XGBRegressor # type: ignore
from sklearn.metrics import mean_squared_error
import joblib

def train_model(X_train, y_train):
    """
    Train an XGBoost regression model.
    """
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using Mean Squared Error (MSE).
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def save_model(model, output_path):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, output_path)