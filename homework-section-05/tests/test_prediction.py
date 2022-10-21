import math
import warnings

import numpy as np

from classification_model.predict import make_prediction

warnings.filterwarnings("ignore")


def test_make_prediction(sample_test_input):
    expected_first_class_pred_val = 0
    expected_first_prob_pred_val = 0.28211531
    expected_num_predictions = 262

    result = make_prediction(sample_test_input)

    preds = result["predictions"]
    probs = result["probabilities"]
    # print(type(preds), preds)
    # print(type(probs), probs)
    assert isinstance(preds, np.ndarray) & isinstance(probs, np.ndarray)
    assert isinstance(preds[0], np.int64) & isinstance(probs[0], np.float64)
    assert result["errors"] is None
    assert expected_num_predictions == len(probs) == len(preds)
    assert math.isclose(probs[0], expected_first_prob_pred_val, abs_tol=0.05)
    assert preds[0] == expected_first_class_pred_val
