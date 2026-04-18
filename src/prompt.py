SYSTEM_INSTRUCTION = """Generate ONLY one line of executable Polars Python code.
`df` is loaded, `pl` is imported. No markdown, no explanation, no print.

Rules:
- Scalars: use .item() after .select() or .sort().head(1)[col].item()
- Row count: .height
- Date methods need parens: .dt.month(), .dt.year()
- Membership: .is_in([...]) not .isin()
- Sort ASCENDING (lowest first) is the default: .sort('col')
- Sort DESCENDING (highest first): .sort('col', descending=True)
- NEVER use ascending=True — that argument does not exist in Polars
- String equality in filter: pl.col('x') == 'value'
"""

FEW_SHOTS = [
    {
        "schema": "df: product (str), revenue (f64), country (str)",
        "question": "What is the total revenue in France?",
        "code": "df.filter(pl.col('country') == 'France').select(pl.col('revenue').sum()).item()",
    },
    {
        "schema": "df: product (str), revenue (f64)",
        "question": "Which product has the highest total revenue?",
        "code": "df.group_by('product').agg(pl.col('revenue').sum()).sort('revenue', descending=True).head(1)['product'].item()",
    },
    {
        "schema": "df: country (str), quantity (i64)",
        "question": "Which country has the lowest average quantity?",
        "code": "df.group_by('country').agg(pl.col('quantity').mean()).sort('quantity').head(1)['country'].item()",
    },
    {
        "schema": "df: date (date), revenue (f64)",
        "question": "Which month number has the highest total revenue?",
        "code": "df.group_by(pl.col('date').dt.month().alias('month')).agg(pl.col('revenue').sum()).sort('revenue', descending=True).head(1)['month'].item()",
    },
    {
        "schema": "df: date (date), amount (f64)",
        "question": "How many rows are in months 1 through 3?",
        "code": "df.filter(pl.col('date').dt.month().is_in([1, 2, 3])).height",
    },
]


def build_prompt(schema: str, question: str) -> str:
    shots = "\n\n".join(
        f"Schema: {s['schema']}\nQ: {s['question']}\nCode: {s['code']}"
        for s in FEW_SHOTS
    )
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Examples:\n{shots}\n\n"
        f"Schema: {schema}\nQ: {question}\nCode:"
    )