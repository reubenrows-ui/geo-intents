from enum import Enum
from pydantic import BaseModel, ConfigDict
from typing import Optional


class RadiusUnit(str, Enum):
    km = "km"
    miles = "miles"
    mi = "mi"
    m = "m"


class SlotCandidates(BaseModel):
    anchor_place: Optional[str]
    radius_value: Optional[float]
    radius_unit: Optional[RadiusUnit]
    entity: Optional[str]
    model_config = ConfigDict(extra='forbid')  # disallow additional properties


class PoiCoords(BaseModel):
    latitude: Optional[float]
    longitude: Optional[float]
    model_config = ConfigDict(extra='forbid') 