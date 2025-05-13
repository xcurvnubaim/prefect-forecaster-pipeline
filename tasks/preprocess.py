from prefect import task
import pandas as pd

@task
def aggregate_sales_data(df, freq: str = "D") -> pd.DataFrame:
    """
    Aggregates sales data by the specified frequency (e.g., daily).
    
    Args:
        df (pd.DataFrame): The DataFrame containing sales data.
        freq (str): The frequency for aggregation (default is 'D' for daily).
    
    Returns:
        pd.DataFrame: The aggregated DataFrame.
    """
    return df.resample(freq).sum()

@task
def handle_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing dates in the DataFrame by reindexing and filling missing values.
    
    Args:
        df (pd.DataFrame): The DataFrame containing sales data.
    
    Returns:
        pd.DataFrame: The DataFrame with missing dates handled.
    """
    # Reindex to include all dates in the range
    all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(all_dates, fill_value=0)
    
    return df

@task
def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows with missing values from the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame containing sales data.
    
    Returns:
        pd.DataFrame: The DataFrame with missing values dropped.
    """
    return df.dropna()