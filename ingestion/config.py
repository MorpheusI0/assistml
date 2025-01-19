from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    MLSEA_SPARQL_ENDPOINT = os.getenv('MLSEA_SPARQL_ENDPOINT')
    MLSEA_USE_CACHE = str(os.getenv('MLSEA_USE_CACHE', False)).lower() in ['true', '1', 't', 'y', 'yes']
    MLSEA_CACHE_DIR = os.path.expanduser(os.getenv('MLSEA_CACHE_DIR', '/tmp/mlsea-cache'))

    MONGO_HOST = os.getenv("MONGO_HOST")
    MONGO_PORT = int(os.getenv("MONGO_PORT"))
    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_PASS = os.getenv("MONGO_PASS")
    MONGO_DB = os.getenv("MONGO_DB", "assistml")

    assert MLSEA_SPARQL_ENDPOINT is not None, "MLSEA_SPARQL_ENDPOINT must be set"
    assert MONGO_HOST is not None, "MONGO_HOST must be set"
    assert MONGO_PORT is not None, "MONGO_PORT must be set"
    assert MONGO_USER is not None, "MONGO_USER must be set"
    assert MONGO_PASS is not None, "MONGO_PASS must be set"
