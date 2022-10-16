import pandas as pd
import numpy as np
from classification_model.processing.features import get_title, get_first_cabin
import joblib
from sklearn.pipeline import Pipeline
#import classification_model.config.config as config
from classification_model.config.core import config, DATA_PATH, TRAINED_MODEL_DIR
from classification_model import __version__ as VERSION
# load the data - it is available open source and online


def load_dataset(*, dataset_path: str) -> pd.DataFrame:
    #
    data = pd.read_csv(dataset_path)
    # replace interrogation marks by NaN values

    data = data.replace('?', np.nan)

    # retain only the first cabin if more than
    # 1 are available per passenger
    data['cabin'] = data['cabin'].apply(get_first_cabin)

    # extracts the title (Mr, Ms, etc) from the name variable
    data['title'] = data['name'].apply(get_title)

    # cast numerical variables as floats
    data['fare'] = data['fare'].astype('float')
    data['age'] = data['age'].astype('float')

    # cast cat variables as strings
    data['pclass'] = data['pclass'].astype(str)

    # drop unnecessary variables
    data.drop(labels=['name','ticket', 'boat', 'body','home.dest'], axis=1, inplace=True)

    # display data
    return data

def remove_old_pipeline():
    pass

def save_pipeline(*, pipeline_to_save : Pipeline) -> None:
    #model_name = f"{config.pipeline_save_file}_{config.VERSION}.pkl"
    model_name = f"{config.app_config.pipeline_save_file}_{VERSION}.pkl"
    file_path = TRAINED_MODEL_DIR / model_name

    remove_old_pipeline()
    joblib.dump(pipeline_to_save, file_path)
