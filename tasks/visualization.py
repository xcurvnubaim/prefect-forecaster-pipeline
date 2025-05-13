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
