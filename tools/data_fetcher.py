import polars as pl
from pathlib import Path
from config import COUNTRY, get_data_path

def get_real_estate_data(limit: int = 500) -> pl.DataFrame:
    """Load and lightly clean real estate data for the selected country."""
    path = get_data_path()

    # Column name variations across datasets – normalize
    column_map = {
        "location": ["area", "neighborhood", "district", "project_name", "location"],
        "price": ["price", "sale_price", "rent_price", "amount"],
        "beds": ["bedrooms", "beds", "no_of_bedrooms"],
        "sqft": ["area_sqft", "size_sqft", "built_up_area", "area"],
    }

    df = pl.read_csv(path, infer_schema_length=10000)

    # Try to rename columns to standard names
    for std_col, possible in column_map.items():
        for col in possible:
            if col in df.columns:
                df = df.rename({col: std_col})
                break

    # Keep only useful columns if they exist
    keep_cols = ["location", "price", "beds", "sqft"]
    existing = [c for c in keep_cols if c in df.columns]
    df = df.select(existing)

    # Basic cleaning
    df = df.filter(pl.col("price").is_not_null())
    if "price" in df.columns:
        df = df.with_columns(pl.col("price").cast(pl.Float64))

    return df.head(limit).lazy()  # lazy for efficiency