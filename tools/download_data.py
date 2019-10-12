#!/usr/bin/env python
import requests, zipfile, io

URL = 'http://www.hlt.utdallas.edu/~vgogate/ml/2019f/homeworks/netflix.zip'
DIR = 'data'

def main():  # noqa
    r = requests.get(URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(DIR)

if __name__ == '__main__':
    main()
