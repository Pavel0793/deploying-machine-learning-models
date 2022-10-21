import warnings

from classification_model import __version__ as VERSION
from classification_model.config.core import config
from classification_model.processing.data_manager import load_pipeline
from classification_model.processing.validation import validate_inputs

warnings.filterwarnings("ignore")


def make_prediction(test_set):
    validated_data, errors = validate_inputs(test_set)
    print("errors", errors)
    model = load_pipeline()
    results = {
        "predictions": None,
        "probabilities": None,
        "version": VERSION,
        "errors": errors,
    }

    if errors is None:
        probs = model.predict_proba(validated_data[config.model_config.features])[:, 1]
        predictions = model.predict(validated_data[config.model_config.features])
        results["predictions"] = predictions
        results["probabilities"] = probs

    return results
