from pydantic import BaseModel


class NumberView(BaseModel):
    number: int
