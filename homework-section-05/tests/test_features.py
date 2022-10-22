import pandas as pd

from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer


def test_extract_letter_transformer(sample_test_input):
    transformer = ExtractLetterTransformer(
        variables=config.model_config.extract_letter_vars
    )

    assert pd.isnull(sample_test_input["cabin"].iat[0])
    assert sample_test_input["cabin"].iat[5] == "G6"

    sample_test_input_transformed = transformer.fit_transform(sample_test_input)

    assert pd.isnull(sample_test_input_transformed["cabin"].iat[0])
    assert sample_test_input_transformed["cabin"].iat[5] == "G"
