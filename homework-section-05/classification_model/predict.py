import warnings
from typing import Union

import pandas as pd

from classification_model import __version__ as VERSION
from classification_model.processing.data_manager import load_pipeline
from classification_model.processing.validation import validate_inputs

warnings.filterwarnings("ignore")
model = load_pipeline()


def make_prediction(*, test_set: Union[pd.DataFrame, dict],
                   ) -> dict:
    test_set = pd.DataFrame(test_set)
    validated_data, errors = validate_inputs(test_set)


    results = {
        "predictions": None,
        "probabilities": None,
        "version": VERSION,
        "errors": errors,

    }

    if errors is None:
        probs = model.predict_proba(validated_data)[:, 1]
        predictions = model.predict(validated_data)
        results["predictions"] = predictions
        results["probabilities"] = probs

    return results
