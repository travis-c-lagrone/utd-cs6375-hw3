#!/usr/bin/env python
import requests, zipfile, io

URL = 'http://www.hlt.utdallas.edu/~vgogate/ml/2019f/homeworks/netflix.zip'
DIR = 'data'

def download_data(url_: str=URL, dir_: str=DIR) -> None:
    """Download and extract a .zip of data from ``url_`` into ``dir_``.

    Args:
        url_ (str): The URL of the zip file to download. Defaults to :py:const:`URL`.
        dir_ (str): The local folder into which to extract the zip. Defaults to :py:const:`DIR`.

    """
    r = requests.get(url_)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dir_)

if __name__ == '__main__':
    download_data()
