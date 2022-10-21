from typing import List, Optional

from pydantic import BaseModel, ValidationError


def validate_inputs(input_data):

    validated_data = input_data.copy()
    errors = None
    try:
        MultipleTitanicDataInputs(inputs=validated_data.to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()
    return validated_data, errors


class TitanicDataInputSchema(BaseModel):
    pclass: Optional[str]
    sex: Optional[str]
    age: Optional[float]
    sibsp: Optional[int]
    parch: Optional[int]
    fare: Optional[float]
    cabin: Optional[str]
    embarked: Optional[str]
    title: Optional[str]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]
