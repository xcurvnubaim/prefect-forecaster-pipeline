from prefect import task
import matplotlib.pyplot as plt

@task
def plot_sales_forecast(df, future_months, predictions):
    plt.plot(df["month"], df["sales"], label="Actual Sales")
    plt.plot([m[0] for m in future_months], predictions, label="Forecast", linestyle="--")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.legend()
    plt.title("Sales Forecast")
    plt.savefig("forecast.png")
    print("Saved forecast to forecast.png")
