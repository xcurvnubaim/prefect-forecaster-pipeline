from tasks import feature_engineering, preprocess
from prefect import flow
import pandas as pd

@flow(name="Feature Engineering Pipeline")
def feature_engineering_pipeline_concurrently(df, target_column: str, lag_days: int, lag_weeks: int, window_size: list[int]):
    """
    Feature engineering pipeline for sales data.
    
    Args:
        df (pd.DataFrame): The DataFrame containing sales data.
        target_column (str): The name of the target column.
        lag_days (int): Number of lag days to create.
        lag_weeks (int): Number of lag weeks to create.
        window_size (list[int]): List of window sizes for rolling features.
    
    Returns:
        pd.DataFrame: The DataFrame with engineered features.
    """

    tasks = [
        feature_engineering.add_lag_features.submit(df, target_column, lag_days, lag_weeks),
        feature_engineering.add_rolling_features.submit(df, target_column, window_size),
        feature_engineering.add_time_features.submit(df),
        feature_engineering.add_ramadhan_features.submit(df),
    ]

    df = df.copy()
    for future in tasks:
        result = future.result()
        df = df.join(result.drop(columns=df.columns, errors='ignore'), how='left')

    return df

# @flow(name="Feature Engineering Pipeline")
def feature_engineering_pipeline_sequential(df, target_column: str, lag_days: int, lag_weeks: int, window_size: list[int]):
    """
    Feature engineering pipeline for sales data.
    
    Args:
        df (pd.DataFrame): The DataFrame containing sales data.
        target_column (str): The name of the target column.
        lag_days (int): Number of lag days to create.
        lag_weeks (int): Number of lag weeks to create.
        window_size (list[int]): List of window sizes for rolling features.
    
    Returns:
        pd.DataFrame: The DataFrame with engineered features.
    """

    lag_df = feature_engineering.add_lag_features(df, target_column, lag_days, lag_weeks)
    rolling_df = feature_engineering.add_rolling_features(df, target_column, window_size)
    time_df = feature_engineering.add_time_features(df)
    ramadhan_df = feature_engineering.add_ramadhan_features(df)
    
    return pd.concat([df, lag_df, rolling_df, time_df, ramadhan_df], axis=1)