from prefect import task
from tasks.feature_engineering import ForecastFeatureEngineering
from datetime import timedelta
import pandas as pd
import joblib

@task
def forecast_future_sales(
        model_path: str, 
        n_days: int, 
        history: pd.DataFrame, 
        target_column: str,
        lag_days: int,
        lag_weeks: int,
        window_size: list[int]
    ) -> pd.DataFrame:
    """
    Forecast future sales using the trained model.
    
    Args:
        model_path (str): Path to the trained model.
        n_days (int): Number of days to forecast.
        history (pd.DataFrame): Historical data for feature generation.
        target_column (str): The name of the target column.
        lag_days (int): Number of lag days for feature generation.
        lag_weeks (int): Number of lag weeks for feature generation.
        window_size (list[int]): List of window sizes for rolling features.
    
    Returns:
        pd.DataFrame: DataFrame containing the forecasted values.
    """


    # Load the trained model
    model = joblib.load(model_path)
    
    # Initialize the feature engineering class
    features = ForecastFeatureEngineering(target_column, lag_days, lag_weeks, window_size)
    
    # Generate future dates
    last_date = history.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
    
    # Create a DataFrame for future dates
    future_df = pd.DataFrame(index=future_dates)

    # Iteratively predict future values for each day
    predictions = []
    for i in range(n_days):
        # Generate features for the current day (taking into account previous data)
        lag_features = features.add_lag_features(history)
        rolling_features = features.add_rolling_features(history)
        time_features = features.add_time_features(history)
        ramadhan_features = features.add_ramadhan_features(history)
        
        # Combine all features into a single DataFrame for prediction
        future_day_features = pd.concat([lag_features, rolling_features, time_features, ramadhan_features], axis=1)
        
        # Make the prediction for the current day
        future_day_prediction = model.predict(future_day_features)
        
        # Store the predicted value
        predictions.append(future_day_prediction[0])
        
        # Append the predicted value to the history DataFrame for the next iteration
        new_row = pd.DataFrame({target_column: future_day_prediction}, index=[future_dates[i]])
        history = pd.concat([history, new_row])

    # Populate the future DataFrame with the predictions
    future_df[target_column] = predictions
    
    return future_df