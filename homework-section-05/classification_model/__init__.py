import warnings

from classification_model.config.core import PACKAGE_ROOT

warnings.simplefilter("ignore", UserWarning)

with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()
