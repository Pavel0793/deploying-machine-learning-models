# for encoding categorical variables
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder

# from feature-engine
# for imputation
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)

# to build the models
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# feature scaling
# for the preprocessors
from sklearn.preprocessing import StandardScaler

# import config.config as config
from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer

# from feature_engine.selection import DropFeatures
# from feature_engine.transformation import LogTransformer
# from feature_engine.wrappers import SklearnTransformerWrapper


# set up the pipeline
titanic_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute categorical variables with string 'missing'
        (
            "categorical_imputation",
            CategoricalImputer(
                imputation_method=config.model_config.imputation_method_cat,
                variables=config.model_config.categorical_variables,
            ),
        ),
        # add missing indicator to numerical variables
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numeric_variables),
        ),
        # impute numerical variables with the median
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method=config.model_config.imputation_method_mean_median,
                variables=config.model_config.numeric_variables_with_na,
            ),
        ),
        # Extract first letter from cabin
        (
            "extract_letter",
            ExtractLetterTransformer(variables=config.model_config.extract_letter_vars),
        ),
        # == CATEGORICAL ENCODING ======
        # remove categories present in less than 5% of the observations (0.05)
        # group them in one category called 'Rare'
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=config.model_config.tol,
                replace_with=config.model_config.replace_with,
                variables=config.model_config.categorical_variables,
            ),
        ),
        # encode categorical variables using one hot encoding into k-1 variables
        (
            "categorical_encoder",
            OneHotEncoder(variables=config.model_config.categorical_variables),
        ),
        # scale using standardization
        ("scaler", StandardScaler()),
        # logistic regression (use C=0.0005 and random_state=0)
        (
            "Logit",
            LogisticRegression(
                C=config.model_config.C, random_state=config.model_config.random_state
            ),
        ),
    ]
)
