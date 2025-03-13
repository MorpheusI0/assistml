import asyncio

from common.data import ObjectDocumentMapper
from mlsea import mlsea_repository as mlsea
from processing.dataset import process_all_datasets

POSSIBLE_ENTITIES = ['dataset', 'task', 'run']

def process_initial_offset(entity: str, entity_id: int):
    offset_ids = {entity: entity_id}
    if 'run' in offset_ids:
        task_id = mlsea.retrieve_task_id_for_run_id(offset_ids['run'])
        offset_ids['task'] = task_id
    if 'task' in offset_ids:
        dataset_id = mlsea.retrieve_dataset_id_for_task_id(offset_ids['task'])
        offset_ids['dataset'] = dataset_id
    return offset_ids

async def main(initial_offset=None, head=None):
    # input validation
    if initial_offset is not None:
        entity, entity_id = initial_offset.split(':')
        if entity not in POSSIBLE_ENTITIES:
            raise ValueError(f'Invalid entity: {entity}. Possible entities: {", ".join(POSSIBLE_ENTITIES)}')
        initial_offset = process_initial_offset(entity, int(entity_id))

    if head is not None:
        if not head.isdigit():
            raise ValueError(f'Invalid head: {head}. Must be a number.')
        head = int(head)

    odm = ObjectDocumentMapper()
    await odm.connect()

    await process_all_datasets(recursive=True, offset=initial_offset, head=head)


if __name__ == '__main__':
    asyncio.run(main())
