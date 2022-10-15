from sklearn.pipeline import Pipeline
# for the preprocessors
from sklearn.preprocessing import MinMaxScaler, Binarizer
# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LogisticRegression
# from feature-engine
# for imputation
from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer,
    CategoricalImputer,
)
# for encoding categorical variables
from feature_engine.encoding import (
    RareLabelEncoder,
    OrdinalEncoder,
    OneHotEncoder
)

from feature_engine.transformation import LogTransformer

from feature_engine.selection import DropFeatures
from feature_engine.wrappers import SklearnTransformerWrapper
from processing.features import ExtractLetterTransformer
import config.config as config

# set up the pipeline
titanic_pipe = Pipeline([

    # ===== IMPUTATION =====
    # impute categorical variables with string 'missing'
    ('categorical_imputation', CategoricalImputer(imputation_method =config.IMPUTATION_METHOD_CAT, variables=config.CATEGORICAL_VARIABLES)),

    # add missing indicator to numerical variables
    ('missing_indicator', AddMissingIndicator(variables = config.NUMERIC_VARIABLES)),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(imputation_method = config.IMPUTATION_METHOD_MEAN_MEDIAN,
                                           variables= config.NUMERIC_VARIABLES_WITH_NA)),


    # Extract first letter from cabin
    ('extract_letter', ExtractLetterTransformer(variables=config.EXTRACT_LETTER_VARS)),


    # == CATEGORICAL ENCODING ======
    # remove categories present in less than 5% of the observations (0.05)
    # group them in one category called 'Rare'
    ('rare_label_encoder', RareLabelEncoder(tol = config.TOL,
                                           replace_with = config.REPLACE_WITH,
                                           variables=config.CATEGORICAL_VARIABLES)),


    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder', OneHotEncoder(variables=config.CATEGORICAL_VARIABLES)),

    # scale using standardization
    ('scaler', StandardScaler()),

    # logistic regression (use C=0.0005 and random_state=0)
    ('Logit', LogisticRegression(C=config.C, random_state=config.RANDOM_STATE)),
])