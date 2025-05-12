from prefect import task
import pandas as pd
import numpy as np

@task
def load_sales_data():
    df = pd.DataFrame({
        "month": np.arange(1, 13),
        "sales": [100, 120, 130, 150, 160, 170, 200, 210, 190, 230, 240, 260]
    })
    return df
