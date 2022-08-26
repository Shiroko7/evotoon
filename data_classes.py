from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class IntParam:
    name: str
    min_val: int
    max_val: int


@dataclass
class FloatParam:
    name: str
    min_val: float
    max_val: float
    decs: int


@dataclass
class CatParam:
    name: str
    values: List
