from prefect import flow
from tasks.data import load_sales_data
from tasks.model import train_sales_model, predict_future_sales
from tasks.visualization import plot_sales_forecast

@flow(name="Sales Forecast Pipeline")
def forecast_pipeline():
    df = load_sales_data()
    model = train_sales_model(df)
    future_months, predictions = predict_future_sales(model)
    plot_sales_forecast(df, future_months, predictions)
