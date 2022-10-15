package_name= 'classification_model'

DATA_PATH = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
TRAINED_MODEL_DIR = "trained_modeles"

# data sources
TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'
# vars and features
TARGET = "survived"

# train test
TEST_SIZE = 0.2
# random_state and other constants
RANDOM_STATE = 0

C = 0.0005
TOL = 0.05
REPLACE_WITH = "Rare"

# pipeline constants
IMPUTATION_METHOD_CAT = "missing"

IMPUTATION_METHOD_MEAN_MEDIAN = "median"

EXTRACT_LETTER_VARS = ["cabin"]

NUMERIC_VARIABLES= [ 'age','sibsp', 'parch','fare']

NUMERIC_VARIABLES_WITH_NA=  ['age', 'fare']

CATEGORICAL_VARIABLES= ['pclass','sex','cabin','embarked', 'title']


CATEGORICAL_VARIABLES_WITH_NA= ['cabin', 'embarked']


