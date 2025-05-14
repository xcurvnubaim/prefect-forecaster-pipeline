from flows.forecast_flow import forecast_pipeline, SalesForecastConfig

if __name__ == "__main__":
    forecast_pipeline(
        config=SalesForecastConfig(
            file_path="data/processed.csv",
            date_column="TANGGAL",
            product_id="MP000197_KD000028_PL000036_SZ000012",
            target_column="BERAT_TOTAL",
            use_test=True,
            end_train="2021-12-31",
            future_days=365,
            lag_days=30,
            lag_weeks=52,
            window_size=[7, 30],
            frequency="D",
        )
    )
