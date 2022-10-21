# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score

# import config.config as config
from classification_model.config.core import DATA_PATH
from classification_model.pipeline import titanic_pipe
from classification_model.processing.data_manager import (
    load_dataset,
    load_train_test,
    save_pipeline,
)


def run_pipeline() -> None:
    # loading
    load_dataset(dataset_path=DATA_PATH)

    X_train, y_train = load_train_test(data_type="train")
    X_test, y_test = load_train_test(data_type="test")
    # pipeline training
    titanic_pipe.fit(X_train, y_train)

    # pipeline saving

    save_pipeline(pipeline_to_save=titanic_pipe)
    # make predictions for train set
    class_ = titanic_pipe.predict(X_train)
    pred = titanic_pipe.predict_proba(X_train)[:, 1]

    # determine mse and rmse
    print("train roc-auc: {}".format(roc_auc_score(y_train, pred)))
    print("train accuracy: {}".format(accuracy_score(y_train, class_)))
    print()

    # make predictions for test set
    class_ = titanic_pipe.predict(X_test)
    pred = titanic_pipe.predict_proba(X_test)[:, 1]

    # determine mse and rmse
    print("test roc-auc: {}".format(roc_auc_score(y_test, pred)))
    print("test accuracy: {}".format(accuracy_score(y_test, class_)))
    print()


if __name__ == "__main__":
    run_pipeline()
