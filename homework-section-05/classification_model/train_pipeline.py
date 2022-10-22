# import config.config as config
from classification_model.config.core import DATA_PATH
from classification_model.pipeline import titanic_pipe
from classification_model.processing.data_manager import (
    load_dataset,
    load_train_test,
    save_pipeline,
)
from classification_model.processing.metrics import calculate_metrics


def run_pipeline() -> None:
    # loading
    load_dataset(dataset_path=DATA_PATH)

    X_train, y_train = load_train_test(data_type="train")
    X_test, y_test = load_train_test(data_type="test")

    # pipeline training
    titanic_pipe.fit(X_train, y_train)

    # pipeline saving
    save_pipeline(pipeline_to_save=titanic_pipe)

    # metrics calculation
    calculate_metrics(titanic_pipe, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    run_pipeline()
