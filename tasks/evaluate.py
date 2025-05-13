from prefect import task
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import joblib

@task
def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Evaluates the model performance using various metrics.
    
    Args:
        y_true (pd.Series): The true target values.
        y_pred (pd.Series): The predicted target values.
    
    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }
