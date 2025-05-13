import csv
from prefect import flow, task
from prefect.artifacts import create_table_artifact

@task
def load_csv_as_table(path: str) -> list[dict[str, str]]:
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)

@flow
def csv_to_table_artifact_flow():
    table_data = load_csv_as_table("data/sales_data.csv")
    create_table_artifact(
        table=table_data,
        key="csv-table",  # must be lowercase letters, numbers, dashes only
        description="CSV rendered as a Prefect table artifact"
    )

if __name__ == "__main__":
    csv_to_table_artifact_flow()
