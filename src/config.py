import os

# Endee configuration
ENDPOINT = os.environ.get("ENDEE_API_URL", "http://localhost:8080")
ENDPOINT_TOKEN = os.environ.get("ENDEE_API_TOKEN", "")

COLLECTION_NAME = os.environ.get("ENDEE_COLLECTION", "stackoverflow")

# Embedding model
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval config
TOP_K = int(os.environ.get("TOP_K", "5"))

# Data paths
RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH", "data/train/Questions.csv")
PROCESSED_DATA_PATH = os.environ.get("PROCESSED_DATA_PATH", "data/processed/sample.csv")

# Sampling
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "5000"))

# OpenAI (optional)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
