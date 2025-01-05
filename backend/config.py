from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = os.getenv("PORT", 8080)
    DEBUG = os.getenv("DEBUG", True)
    VERBOSE = os.getenv("VERBOSE", True)
    WORKING_DIR = os.getenv("WORKING_DIR", "~/.assistml/working")
    MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
    MONGO_PORT = os.getenv("MONGO_PORT", 27017)
    MONGO_USER = os.getenv("MONGO_USER", "admin")
    MONGO_PASS = os.getenv("MONGO_PASS", "admin")
