from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    @staticmethod
    def _parse_bool(value):
        return str(value).lower() in ['true', '1', 't', 'y', 'yes']

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = os.getenv("PORT", 8080)
    DEBUG = _parse_bool(os.getenv("DEBUG", True))
    VERBOSE = _parse_bool(os.getenv("VERBOSE", True))

    WORKING_DIR = os.getenv("WORKING_DIR", "~/.assistml/working")
    SAVE_UPLOADS = _parse_bool(os.getenv("SAVE_UPLOADS", False))

    MONGO_HOST = os.getenv("MONGO_HOST")
    MONGO_PORT = int(os.getenv("MONGO_PORT"))
    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_PASS = os.getenv("MONGO_PASS")
    MONGO_DB = os.getenv("MONGO_DB", "assistml")
    MONGO_TLS = _parse_bool(os.getenv("MONGO_TLS", False))

    assert MONGO_HOST is not None, "MONGO_HOST must be set"
    assert MONGO_PORT is not None, "MONGO_PORT must be set"
    assert MONGO_USER is not None, "MONGO_USER must be set"
    assert MONGO_PASS is not None, "MONGO_PASS must be set"


