import pytest

from classification_model.processing.data_manager import load_train_test


@pytest.fixture()
def sample_test_input():
    X_test, y_test = load_train_test(data_type="test")
    return X_test
