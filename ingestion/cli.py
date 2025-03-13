import click
import asyncio
from main import POSSIBLE_ENTITIES, main as ingestion_main

def validate_offset(ctx, param, value):
    if value is not None:
        entity, entity_id = value.split(':')
        if entity not in POSSIBLE_ENTITIES:
            raise click.BadParameter(f'Invalid entity: {entity}. Possible entities: {", ".join(POSSIBLE_ENTITIES)}')
        if not entity_id.isdigit():
            raise click.BadParameter(f'Invalid ID: {entity_id}. Must be a number.')
    return value

@click.command()
@click.option('--offset', default=None, callback=validate_offset, help=f'Initial offset to start processing with. Format: <entity>:<ID>, e.g. dataset:1. Possible entities: {", ".join(POSSIBLE_ENTITIES)}')
@click.option('--head', default=None, help='Number of datasets to process')
def main(offset, head):
    click.echo(f"Starting ingestion with offset {offset} and head {head}")
    asyncio.run(ingestion_main(offset, head))

if __name__ == '__main__':
    main()
