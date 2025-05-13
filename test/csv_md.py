import csv
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

@task
def csv_to_markdown_table(csv_path: str) -> str:
    
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    headers = rows[0]
    table = f"| {' | '.join(headers)} |\n"
    table += f"| {' | '.join(['---'] * len(headers))} |\n"
    for row in rows[1:]:
        table += f"| {' | '.join(row)} |\n"

    return table

@flow
def csv_report_flow():
    markdown_table = csv_to_markdown_table("data/sales_data.csv")
    create_markdown_artifact(
        key="csv-markdown-table",
        markdown=markdown_table,
        description="CSV data shown as Markdown table"
    )

if __name__ == "__main__":
    csv_report_flow()
