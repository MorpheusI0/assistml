from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from config import Config
from .model import Model
from .dataset import Dataset
from .implementation import Implementation
from .query import Query
from .task import Task
from .enriched_model import EnrichedModel


class ObjectDocumentMapper:
    def __init__(self, host: str = Config.MONGO_HOST, port: int = Config.MONGO_PORT, username: str = Config.MONGO_USER,
                 password: str = Config.MONGO_PASS, db: str = Config.MONGO_DB, tls: bool = Config.MONGO_TLS):
        self._client = AsyncIOMotorClient(
            host=host,
            port=port,
            username=username,
            password=password,
            authSource=db,
            tls=tls
        )
        self._db = self._client[db]

    async def connect(self):
        await init_beanie(
            database=self._db,
            document_models=[Dataset, Task, Implementation, Model, Query, EnrichedModel]
        )
