from processing.data_manager import load_dataset
import config.config as config
from sklearn.model_selection import train_test_split
from pipeline import titanic_pipe
# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score

def run_pipeline():
    # loading
    data = load_dataset(dataset_path = config.DATA_PATH)

    # train test split
    X = data.drop('survived', axis = 1)
    y = data['survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size= config.TEST_SIZE,
        random_state=config.RANDOM_STATE)

    # pipeline training
    titanic_pipe.fit(X_train, y_train)

    # pipeline saving
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
