import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def get_title(passenger):
    line = passenger
    if re.search("Mrs", line):
        return "Mrs"
    elif re.search("Mr", line):
        return "Mr"
    elif re.search("Miss", line):
        return "Miss"
    elif re.search("Master", line):
        return "Master"
    else:
        return "Other"


def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan


class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError("Variables should be list")
        self.variables = variables

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X = X.copy()

        for var in self.variables:
            X[var] = X[var].apply(lambda x: x[0] if type(x) == str else x)

        return X
