import asyncio

from common.data import ObjectDocumentMapper
from processing.dataset import process_all_datasets


async def main():
    odm = ObjectDocumentMapper()
    await odm.connect()

    await process_all_datasets(recursive=True, head=10)


if __name__ == '__main__':
    asyncio.run(main())
