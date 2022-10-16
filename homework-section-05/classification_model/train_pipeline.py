from processing.data_manager import load_dataset, save_pipeline
#import config.config as config
from classification_model.config.core import config, DATA_PATH
from sklearn.model_selection import train_test_split
from pipeline import titanic_pipe
# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score

def run_pipeline() -> None:
    # loading
    data = load_dataset(dataset_path = DATA_PATH)

    # train test split
    X = data[config.model_config.features]
    y = data[config.model_config.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size= config.model_config.test_size,
        random_state=config.model_config.random_state)

    # pipeline training
    titanic_pipe.fit(X_train, y_train)

    # pipeline saving

    save_pipeline(pipeline_to_save=titanic_pipe)
    # make predictions for train set
    class_ = titanic_pipe.predict(X_train)
    pred = titanic_pipe.predict_proba(X_train)[:, 1]

    # determine mse and rmse
    print('train roc-auc: {}'.format(roc_auc_score(y_train, pred)))
    print('train accuracy: {}'.format(accuracy_score(y_train, class_)))
    print()

    # make predictions for test set
    class_ = titanic_pipe.predict(X_test)
    pred = titanic_pipe.predict_proba(X_test)[:, 1]

    # determine mse and rmse
    print('test roc-auc: {}'.format(roc_auc_score(y_test, pred)))
    print('test accuracy: {}'.format(accuracy_score(y_test, class_)))
    print()

if __name__ == '__main__':
    run_pipeline()
