from typing import List

from pydantic import BaseModel

from common.data.query import Report


class ReportResponseDto(Report):
    pass
