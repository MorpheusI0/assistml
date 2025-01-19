import asyncio

from mlsea import MLSeaRepository
from common.data import ObjectDocumentMapper
from processing.dataset import DatasetProcessor


async def main():
    mlsea = MLSeaRepository()
    odm = ObjectDocumentMapper()
    await odm.connect()

    dataset_processor = DatasetProcessor(mlsea)
    await dataset_processor.process(recursive=True, head=3)


if __name__ == '__main__':
    asyncio.run(main())
