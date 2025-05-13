from prefect import task
import pandas as pd
from util.ramadhan import is_ramadhan_day

@task
def add_lag_features(df: pd.DataFrame, target_column: str, lag_days: int, lag_weeks: int) -> pd.DataFrame:
    lags = {
        f"lag_{i}": df[target_column].shift(i)
        for i in range(1, lag_days + 1)
    }
    week_lags = {
        f"lag_week_{i}": df[target_column].shift(i * 7)
        for i in range(1, lag_weeks + 1)
    }
    return pd.DataFrame({**lags, **week_lags})


@task
def add_rolling_features(df: pd.DataFrame, target_column: str, window_sizes: list[int]) -> pd.DataFrame:
    rolling_features = {
        f"rolling_mean_{w}": df[target_column].rolling(window=w).mean()
        for w in window_sizes
    }
    rolling_features.update({
        f"rolling_std_{w}": df[target_column].rolling(window=w).std()
        for w in window_sizes
    })
    return pd.DataFrame(rolling_features)


@task
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    index = df.index
    return pd.DataFrame({
        "day_of_week": index.dayofweek,
        "month": index.month,
        "year": index.year,
        "day_of_year": index.dayofyear,
        # "week_of_year": index.isocalendar().week if needed
    }, index=index)


@task
def add_ramadhan_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "is_ramadhan": df.index.to_series().map(is_ramadhan_day)
    }, index=df.index)