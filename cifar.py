import os
import tarfile

import requests

__cifar_10_url = "http://www.cs.utoronto.ca/%7Ekriz/cifar-10-binary.tar.gz"
__cifar_100_url = "http://www.cs.utoronto.ca/%7Ekriz/cifar-100-binary.tar.gz"


def prepare_cifar_10():
    path = os.path.basename(__cifar_10_url)
    if os.path.exists(path):
        return
    __download_file(__cifar_10_url, path)
    __extract_file(path)


def prepare_cifar_100():
    path = os.path.basename(__cifar_100_url)
    if os.path.exists(path):
        return
    __download_file(__cifar_100_url, path)
    __extract_file(path)


def __extract_file(path):
    print("Extracting %s..." % path)
    with tarfile.open(path) as tar:
        tar.extractall()


def __download_file(url, path):
    print("Downloading %s..." % url)
    r = requests.get(url)
    if r.status_code == 200:
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
    else:
        raise IOError("Error downloading %s" % url)
