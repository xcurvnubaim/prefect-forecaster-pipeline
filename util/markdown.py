import csv

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