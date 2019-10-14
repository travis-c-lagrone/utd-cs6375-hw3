"""Functions to download and prepare data."""

from collections import OrderedDict
from enum import Enum
from pathlib import Path

import pandas as pd
import requests
import zipfile
import io


_DATA_DIR = Path.cwd().joinpath('data')
"""Path: The path of directory into which to extract the zip file.

   Defaults to ``$PWD/data``.
"""


_ZIP_URL = 'http://www.hlt.utdallas.edu/~vgogate/ml/2019f/homeworks/netflix.zip'
"""str: The URL of the zip file containing the modified Netflix dataset against which to test."""

def download_data(path: Path=_DATA_DIR, *, url: str=_ZIP_URL) -> Path:
    """Download and extract a .zip of data from ``url_`` into ``dir_``.

    Args:
        path (str): The directory into which to extract the zip.
            Defaults to :py:const:`DATA_DIR`.
        url (str): The URL of the zip file to download.
            Defaults to :py:const:`ZIP_URL`.
            Keyword only.

    Returns:
        Path: The directory into which the downloaded zip is extracted.

    """
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)
    return path


_DATA_SCHEMA = OrderedDict(MovieID=int, UserID=int, Rating=float)
_TRAIN_FILENAME = 'TrainingRatings.txt'
_TEST_FILENAME = 'TestingRatings.txt'

class DataSplit(Enum):  # noqa: D101
    TEST = _TEST_FILENAME
    TRAIN = _TRAIN_FILENAME

    def __init__(self, filename):  # noqa: D107
        self.filename = filename

def import_data(split: DataSplit, *, path: Path=_DATA_DIR) -> pd.DataFrame:  # noqa: D103
    fp = path.joinpath(split.filename)
    df = pd.read_csv(fp, header=None, names=_DATA_SCHEMA.keys(), dtype=_DATA_SCHEMA)
    return df


_METADATA_SCHEMA = OrderedDict(MovieID=int, YearOfRelease=int, Title=str)
_METADATA_FILENAME = 'movie_titles.txt'

def import_metadata(*, path: Path=_DATA_DIR) -> pd.DataFrame:  # noqa: D103
    fp = path.joinpath(_METADATA_FILENAME)
    df = pd.read_csv(fp, header=None, columns=_METADATA_SCHEMA.keys(), dtype=_METADATA_SCHEMA)
    return df


if __name__ == '__main__':
    download_data()
