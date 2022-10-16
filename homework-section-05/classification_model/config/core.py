import classification_model
from pathlib import Path
from typing import Union, Dict, List, Sequence
from strictyaml import YAML, load
from pydantic import BaseModel
# links
PACKAGE_ROOT = Path(classification_model.__file__).resolve().parent
CONFIGE_FILE_PATH = PACKAGE_ROOT / "config.yml"
ROOT = PACKAGE_ROOT.parent
DATA_PATH = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

class ModelConfig(BaseModel):
    train_data_file : str
    test_data_file : str
    target : str
    test_size : float
    random_state : int
    C : Union[int, float]
    tol : float
    replace_with : str
    imputation_method_cat : str
    imputation_method_mean_median : str
    extract_letter_vars : List[str]
    numeric_variables : List[str]
    numeric_variables_with_na : List[str]
    categorical_variables : List[str]
    categorical_variables_with_na : List[str]
    features : List[str]

class AppConfig(BaseModel):
    package_name: str
    train_data_file: str
    test_data_file: str
    pipeline_save_file :str

class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig



def find_config():

    if CONFIGE_FILE_PATH.is_file():
        return CONFIGE_FILE_PATH

def load_parse_config(config_path : Path = None) -> YAML:
    """Load YAMl
    parsed_config - >
    YAML({'package_name': 'classification_model',
    'train_data_file': 'train.csv',
    'test_data_file': 'test.csv',
    'target': 'survived', 'test_size': '0.2',
     'random_state': '0','imputation_method_cat': 'missing',
     'imputation_method_mean_median': 'median', 'extract_letter_vars': ['cabin']


     parsed_config.data -> Dict
     {'package_name': 'classification_model',
     'train_data_file': 'train.csv',
     'test_data_file': 'test.csv',
     'target': 'survived',
     'test_size': '0.2',
     'random_state': '0',
     'C': '0.005',
     'tol': '0.05',
     'replace_with': 'Rare',}
     """
    if config_path is None:
        config_path = find_config()

    with open(config_path,'r') as file:
        parsed_config = load(file.read())
        #print('parsed_config',parsed_config)
        return parsed_config

def create_config(parsed_config: YAML = None) -> Config:

    """Pydantic var with methods (constants)
    config.train_data_file -> 'train.csv'
    """
    if parsed_config is None:
        parsed_config = load_parse_config()
    #print(parsed_config)
    _config = Config(
        app_config = AppConfig(**parsed_config.data),
        model_config = ModelConfig(**parsed_config.data)
    )

    return _config

config = create_config()