from pydantic import BaseModel
from pydantic.alias_generators import to_camel


def alias_generator(field_name: str) -> str:
    if field_name == "id":
        return "_id"

    if field_name.startswith("_"):
        return field_name

    if field_name in ["revision_id"]:
        return field_name

    return to_camel(field_name)

class CustomBaseModel(BaseModel):

    class Config:
        ser_json_inf_nan = 'constants'
        populate_by_name = True
        alias_generator = alias_generator
