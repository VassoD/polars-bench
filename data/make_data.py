import polars as pl
import random
from datetime import date, timedelta

random.seed(42)
products = ["Widget", "Gadget", "Gizmo", "Thingamajig"]
countries = ["France", "Germany", "Spain", "Italy", "UK"]

rows = []
for i in range(1000):
    rows.append({
        "id": i,
        "product": random.choice(products),
        "revenue": round(random.uniform(10, 500), 2),
        "country": random.choice(countries),
        "date": date(2024, 1, 1) + timedelta(days=random.randint(0, 365)),
        "quantity": random.randint(1, 50),
    })

df = pl.DataFrame(rows)
df.write_parquet("data/sales.parquet")
print(df.head())
print(f"Rows: {df.height}")