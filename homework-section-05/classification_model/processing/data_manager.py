import joblib
import numpy as np
import pandas as pd

# load the data - it is available open source and online
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from classification_model import __version__ as VERSION

# import classification_model.config.config as config
from classification_model.config.core import DATASET_FOLDER, TRAINED_MODEL_DIR, config
from classification_model.processing.features import get_first_cabin, get_title


def split_save_train_test(data: pd.DataFrame) -> None:
    # train test split
    X = data[config.model_config.features]
    y = data[config.model_config.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    train_data = X_train.copy()
    train_data[config.model_config.target] = y_train

    test_data = X_test.copy()
    test_data[config.model_config.target] = y_test

    train_data.to_csv(DATASET_FOLDER / config.app_config.train_data_file, index=False)
    test_data.to_csv(DATASET_FOLDER / config.app_config.test_data_file, index=False)


def load_dataset(*, dataset_path: str) -> pd.DataFrame:
    #
    data = pd.read_csv(dataset_path)
    # replace interrogation marks by NaN values

    data = data.replace("?", np.nan)

    # retain only the first cabin if more than
    # 1 are available per passenger
    data["cabin"] = data["cabin"].apply(get_first_cabin)

    # extracts the title (Mr, Ms, etc) from the name variable
    data["title"] = data["name"].apply(get_title)

    # drop unnecessary variables
    data.drop(
        labels=["name", "ticket", "boat", "body", "home.dest"], axis=1, inplace=True
    )

    # display data
    split_save_train_test(data)

    return data


def remove_old_pipeline():
    pass


def save_pipeline(*, pipeline_to_save: Pipeline) -> None:
    # model_name = f"{config.pipeline_save_file}_{config.VERSION}.pkl"
    model_name = f"{config.app_config.pipeline_save_file}_{VERSION}.pkl"
    file_path = TRAINED_MODEL_DIR / model_name

    remove_old_pipeline()
    joblib.dump(pipeline_to_save, file_path)


def load_pipeline():
    model_name = f"{config.app_config.pipeline_save_file}_{VERSION}.pkl"
    file_path = TRAINED_MODEL_DIR / model_name

    model = joblib.load(file_path)
    return model


def load_train_test(data_type: str):

    if data_type == "train":
        loading_path = DATASET_FOLDER / config.app_config.train_data_file
    elif data_type == "test":
        loading_path = DATASET_FOLDER / config.app_config.test_data_file
    df = pd.read_csv(loading_path)
    # cast numerical variables as floats
    df["fare"] = df["fare"].astype("float")
    df["age"] = df["age"].astype("float")

    # cast cat variables as strings
    df["pclass"] = df["pclass"].astype(str)

    X = df[config.model_config.features]
    y = df[config.model_config.target]

    return (
        X,
        y,
    )
