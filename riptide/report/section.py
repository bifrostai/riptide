from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, root_validator


class ContentType(str, Enum):
    OVERVIEW = "overview"
    TEXT = "text"
    RECALL = "recall"
    AR_SIZE = "ar_size"
    IMAGES = "images"
    PLOT = "plot"


class Content(BaseModel):
    type: ContentType
    header: Optional[str]
    description: Optional[str]
    content: Union[dict, list]
    data: Optional[dict] = dict()

    class Config:
        arbitrary_types_allowed = True

    def tolist(self) -> list:
        return [self.type, self.header, self.description, self.content]


class Section(BaseModel):
    id: str
    title: str = Field(title="Section title")
    subtitle: Optional[str] = Field(title="Section subtitle")
    description: Optional[str] = Field(title="Section description")
    contents: List[Content] = Field(title="Section content")

    def tolist(self) -> list:
        return [
            self.id,
            self.title,
            self.subtitle,
            self.description,
            [c.tolist() for c in self.contents],
        ]

    @root_validator(pre=True)
    def validate(cls, values: dict):
        if "id" not in values:
            return values
        if "title" not in values:
            values["title"] = values["id"]

        return values
