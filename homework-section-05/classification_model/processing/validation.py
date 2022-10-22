from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config


def drop_na_observations_in_test_data(input_data: pd.DataFrame) -> pd.DataFrame:
    input_data = input_data.copy()
    all_possible_na_features = (
        config.model_config.categorical_variables_with_na
        + config.model_config.numeric_variables_with_na
    )
    strange_na_features = [
        col
        for col in config.model_config.features
        if (input_data[col].isnull().sum() > 0) and col not in all_possible_na_features
    ]
    validated_data = input_data.drop(strange_na_features, axis=1)
    return validated_data


def validate_inputs(input_data):

    input_data = input_data[config.model_config.features].copy()

    validated_data = drop_na_observations_in_test_data(input_data)

    errors = None
    try:
        MultipleTitanicDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
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
