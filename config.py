import os
from dotenv import load_dotenv
load_dotenv()

# === CHANGE THESE TO SWITCH EVERYTHING ===
COUNTRY = os.getenv("COUNTRY", "UAE")          # "UAE" | "KSA" | "INDIA"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" | "bedrock" | "snowflake_cortex"
DATA_BACKEND = os.getenv("DATA_BACKEND", "local_polars")  # "local_polars" | "aws_s3" | "snowflake"

# Country-specific data paths (add real CSVs later)
DATA_PATHS = {
    "UAE": "data/dubai_pulse_sample.csv",
    "KSA": "data/aqar_ksa_sample.csv",
    "INDIA": "data/india_rera_sample.csv"
}

# Future Bedrock/Snowflake keys will go here
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")