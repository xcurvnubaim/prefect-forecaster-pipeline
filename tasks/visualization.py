from prefect import task
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

@task
def plot_sales_forecast(y_pred, y_test, target_column: str, index: pd.DatetimeIndex):
    """
    Plots the predicted and actual sales data.
    
    Args:
        y_pred (pd.Series): The predicted sales data.
        y_test (pd.Series): The actual sales data.
    """
     # Convert to Series if needed
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred, index=index, name=target_column)
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test, index=index, name=target_column)

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual Sales', color='blue')
    plt.plot(y_pred.index, y_pred.values, label='Predicted Sales', color='red')
    plt.title('Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid()
    plt.savefig('output/sales_forecast.png')
    plt.close()

def plot_sales_forecast_future(y_pred, target_column: str, index: pd.DatetimeIndex, agg='sum'):
    """
    Plots the predicted future sales data, aggregated to weekly frequency.
    
    Args:
        y_pred (Union[np.ndarray, pd.Series]): The predicted future sales data.
        target_column (str): Name of the target column.
        index (pd.DatetimeIndex): Index representing future dates.
        agg (str): Aggregation method ('mean' or 'sum').
    """
    # Convert to Series if needed
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred, index=index, name=target_column)
    else:
        y_pred = y_pred.copy()
        y_pred.index = index
        y_pred.name = target_column

    # Resample to weekly frequency using the specified aggregation method
    if agg == 'sum':
        y_pred_weekly = y_pred.resample('W').sum()
    else:
        y_pred_weekly = y_pred.resample('W').mean()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_pred_weekly.index, y_pred_weekly.values, label='Predicted Future Sales (Weekly)', color='green')
    plt.title('Future Sales Forecast (Weekly Aggregated)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('output/future_sales_forecast.png')
    plt.close()