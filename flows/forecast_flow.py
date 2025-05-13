from prefect import flow
import tasks
from dataclasses import dataclass
from flows.feature_engineering import feature_engineering_pipeline_concurrently, feature_engineering_pipeline_sequential
import tasks.data
import tasks.evaluate
import tasks.forecast
import tasks.model
import tasks.preprocess
import tasks.visualization

@dataclass
class SalesForecastConfig:
    file_path: str
    date_column: str
    product_id: str
    target_column: str
    end_train: str
    lag_days: int = 30
    lag_weeks: int = 4
    window_size: list[int] = (7, 14, 30)
    frequency: str = "D"
    future_days: int = 365
    

@flow(name="Sales Forecast Pipeline")
def forecast_pipeline(config: SalesForecastConfig):
    df = tasks.data.load_sales_data(config.file_path, config.date_column)
    df_filtered = tasks.data.filter_products(df, config.product_id, config.target_column)
    df_filtered = tasks.preprocess.aggregate_sales_data(df_filtered, config.frequency)
    df_filtered = tasks.preprocess.handle_missing_dates(df_filtered)
    df_filtered = feature_engineering_pipeline_sequential(df_filtered, config.target_column, config.lag_days, config.lag_weeks, config.window_size)
    # df_filtered.to_csv("output/test.csv", index=True)
    df_cleaned = tasks.preprocess.drop_missing_values(df_filtered)
    X_train, y_train, X_test, y_test = tasks.data.split_data(df_cleaned, config.end_train, config.target_column)
    model_paths = tasks.model.train_base_model(X_train, y_train)
    em_model_path = tasks.model.train_em_ensemble(model_paths, X_train, y_train)
    y_pred = tasks.model.predict_em_ensemble(em_model_path, X_test)
    evaluation_results = tasks.evaluate.evaluate_model(y_test, y_pred)
    print(evaluation_results)
    tasks.visualization.plot_sales_forecast(y_test, y_pred, config.target_column, X_test.index) 
    future_sales = tasks.forecast.forecast_future_sales(
        model_path=em_model_path,
        n_days=config.future_days,
        history=df_filtered,
        target_column=config.target_column,
        lag_days=config.lag_days,
        lag_weeks=config.lag_weeks,
        window_size=config.window_size
    )
    future_sales.to_csv("output/future_sales.csv", index=True)
    tasks.visualization.plot_sales_forecast_future(future_sales, config.target_column, future_sales.index)