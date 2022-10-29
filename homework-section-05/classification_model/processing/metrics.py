# to evaluate the models
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def calculate_metrics(titanic_pipe, X_train, X_test, y_train, y_test) -> None:

    # make predictions for train set
    class_ = titanic_pipe.predict(X_train)
    pred = titanic_pipe.predict_proba(X_train)[:, 1]

    # determine accuracy and roc auc
    logger.info("train roc-auc: {}".format(roc_auc_score(y_train, pred)))
    logger.info("train accuracy: {}".format(accuracy_score(y_train, class_)))
    logger.info("classification report" + "\n" + classification_report(y_train, class_))

    # make predictions for test set
    class_ = titanic_pipe.predict(X_test)
    pred = titanic_pipe.predict_proba(X_test)[:, 1]

    # determine accuracy and roc auc
    logger.info("test roc-auc: {}".format(roc_auc_score(y_test, pred)))
    logger.info("test accuracy: {}".format(accuracy_score(y_test, class_)))
    logger.info("classification report" + "\n" + classification_report(y_test, class_))
