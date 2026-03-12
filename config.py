import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# === CHANGE THESE TO SWITCH EVERYTHING ===
COUNTRY = os.getenv("COUNTRY", "UAE")          # "UAE" | "KSA" | "INDIA"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" | "bedrock" | "snowflake_cortex"
DATA_BACKEND = os.getenv("DATA_BACKEND", "local_polars")  # "local_polars" | "aws_s3" | "snowflake"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


# Real paths – relative to project root
DATA_DIR = Path("data")
DATA_PATHS = {
    "UAE": DATA_DIR / "dubai_property_transactions_sample.csv",
    "KSA": DATA_DIR / "aqar_ksa_sample.csv",
    "INDIA": DATA_DIR / "india_rera_sample.csv",
}

def get_data_path():
    path = DATA_PATHS.get(COUNTRY)
    if path and path.exists():
        return path
    raise FileNotFoundError(f"No data file for {COUNTRY}. Checked: {path}")