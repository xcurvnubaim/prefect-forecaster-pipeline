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
        f"rolling_mean_{w}": df[target_column].shift(1).rolling(window=w).mean()
        for w in window_sizes
    }
    # rolling_features.update({
    #     f"rolling_std_{w}": df[target_column].rolling(window=w).std()
    #     for w in window_sizes
    # })
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


class ForecastFeatureEngineering:
    def __init__(self, target_col: str, lag_days: int, lag_weeks: int, window_size: list[int]):
        self.lag_days = lag_days
        self.lag_weeks = lag_weeks
        self.window_size = window_size
        self.target_column = target_col

    def add_lag_features(self, history_df: pd.DataFrame) -> pd.DataFrame:
        # Generate lag features for the last record using shift
        lags = {
            f"lag_{i}": history_df[self.target_column].iloc[-i]  # Shift and get the last value
            for i in range(1, self.lag_days + 1)
        }

        week_lags = {
            f"lag_week_{i}": history_df[self.target_column].iloc[-i*7]  # Shift by week (7 days)
            for i in range(1, self.lag_weeks + 1)
        }

        # Combine all lag features into a single DataFrame for the last record
        lag_features = {**lags, **week_lags}

        return pd.DataFrame([lag_features])
    
    def add_rolling_features(self, history_df: pd.DataFrame) -> pd.DataFrame:
        # Total number of rows
        # total_rows = len(history_df)

        # Generate rolling features for the last record using iloc
        rolling_features = {
            f"rolling_mean_{w}": history_df[self.target_column].iloc[-w:].mean()
            for w in self.window_size
        }
        # print(rolling_features)
        # rolling_features.update({
        #     f"rolling_std_{w}": history_df[self.target_column].iloc[total_rows - w:total_rows].std()
        #     for w in self.window_size
        # })

        return pd.DataFrame([rolling_features])
    
    def add_time_features(self, history_df: pd.DataFrame) -> pd.DataFrame:
        # Get the last record's index
        last_index = history_df.index[-1]

        # Generate time features for the last record only
        time_features = {
            "day_of_week": last_index.dayofweek,
            "month": last_index.month,
            "year": last_index.year,
            "day_of_year": last_index.dayofyear,
        }

        return pd.DataFrame([time_features])
    
    def add_ramadhan_features(self, history_df: pd.DataFrame) -> pd.DataFrame:
        # Get the last record's index
        last_index = history_df.index[-1]
        
        # Generate Ramadhan feature for the last record only
        ramadhan_feature = {
            "is_ramadhan": is_ramadhan_day(last_index)
        }

        return pd.DataFrame([ramadhan_feature])

