import polars as pl
from pathlib import Path
from config import get_data_path, COUNTRY

def get_real_estate_data(limit: int = 500) -> pl.LazyFrame:
    """Load real estate data lazily, with robust column normalization."""
    path = get_data_path()

    df = pl.scan_csv(path, infer_schema_length=100000)

    # More comprehensive mapping (case-insensitive check)
    rename_map = {}
    columns_lower = {c.lower(): c for c in df.columns}

    # Price columns
    price_candidates = ["price", "trans_value", "sale_price", "amount", "transvalue", "value", "saleamount"]
    for cand in price_candidates:
        if cand in columns_lower:
            rename_map[columns_lower[cand]] = "price"
            break

    # Location columns
    loc_candidates = ["area", "neighborhood", "district", "area_en", "project_en", "master_project_en", "location", "project_name", "city"]
    for cand in loc_candidates:
        if cand in columns_lower:
            rename_map[columns_lower[cand]] = "location"
            break

    # Beds / rooms
    beds_candidates = ["bedrooms", "rooms_en", "beds", "no_of_bedrooms"]
    for cand in beds_candidates:
        if cand in columns_lower:
            rename_map[columns_lower[cand]] = "beds"
            break

    # Sqft / area
    sqft_candidates = ["actual_area", "procedure_area", "size", "area_sqft", "built_up_area", "area", "project_area_(sqmts)"]
    for cand in sqft_candidates:
        if cand in columns_lower:
            rename_map[columns_lower[cand]] = "sqft"
            break

    if rename_map:
        df = df.rename(rename_map)

    # Select only what we might use (safe if columns missing)
    possible_cols = ["location", "price", "beds", "sqft"]
    existing = [c for c in possible_cols if c in df.columns]
    if not existing:
        # Fallback to all columns if nothing matched
        df = df.select(pl.all())
    else:
        df = df.select(existing)

    # Defer filter until after collect (avoid lazy schema crash)
    return df.head(limit)  # keep lazy, but head is safe