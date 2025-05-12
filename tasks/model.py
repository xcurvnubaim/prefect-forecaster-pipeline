from prefect import task
from sklearn.linear_model import LinearRegression

@task
def train_sales_model(df):
    model = LinearRegression()
    X = df[["month"]]
    y = df["sales"]
    model.fit(X, y)
    return model

@task
def predict_future_sales(model, months=3):
    future_months = [[i] for i in range(13, 13 + months)]
    predictions = model.predict(future_months)
    return future_months, predictions
