import classification_model
from pathlib import Path

from strictyaml import YAML, load
from pydantic import BaseModel
# links
PACKAGE_ROOT = Path(classification_model.__file__).resolve().parent
CONFIGE_FILE_PATH = PACKAGE_ROOT / "config.yml"
ROOT = PACKAGE_ROOT.parent
DATA_PATH = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_modeles"

class ModelConfig(BaseModel):
    pass
class Config(BaseModel):
    model_config = ModelConfig



def fetch_config_from_yaml():
    pass

def create_and_validate_config(parsed_config: YAML = None) -> Config:
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    #_config = Config(model)