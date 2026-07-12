from typing import Annotated, Literal
from pydantic import BaseModel, Field, field_validator


class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=150, description="Age of the user")]
    sex: Annotated[Literal["male", "female"], Field(..., description="Gender of the user")]
    bmi: Annotated[float, Field(..., gt=0, lt=50, description="BMI of the user")]
    children: Annotated[int, Field(..., ge=0, description="Number of children")]
    smoker: Annotated[Literal["yes", "no"], Field(..., description="Whether the user is a smoker")]
    region: Annotated[
        Literal["southwest", "southeast", "northwest", "northeast"],
        Field(..., description="Region of the user")
    ]

    @field_validator("sex", "smoker", "region", mode="before")
    @classmethod
    def normalize_strings(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value