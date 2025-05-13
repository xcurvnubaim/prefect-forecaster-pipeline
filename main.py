from flows.forecast_flow import forecast_pipeline, SalesForecastConfig

if __name__ == "__main__":
    forecast_pipeline(
        config=SalesForecastConfig(
            file_path="data/processed.csv",
            date_column="TANGGAL",
            product_id="MP000294_KD000016_PL000037_SZ000012",
            target_column="BERAT_TOTAL",
            end_train="2021-12-31",
            lag_days=30,
            lag_weeks=4,
            window_size=[7, 14, 30],
            frequency="D",
        )
    )
