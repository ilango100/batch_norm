import os
import tarfile

import numpy as np
import requests

__cifar_10_url = "http://www.cs.utoronto.ca/%7Ekriz/cifar-10-binary.tar.gz"
__cifar_100_url = "http://www.cs.utoronto.ca/%7Ekriz/cifar-100-binary.tar.gz"

cifar10_dir = "cifar-10-batches-bin"
batch_size = 5000
epochs = 2


def cifar10_labels():
    return open(os.path.join(
        cifar10_dir, "batches.meta.txt"), "r").read().splitlines()


def cifar10_train():
    data_batch = os.path.join(cifar10_dir, "data_batch_%d.bin")
    trim, trlab = [], []

    for _ in range(epochs):
        for i in range(1, 6):
            with open(data_batch % i, "rb") as f:
                for _ in range(10000):
                    trlab.append(int.from_bytes(f.read(1), 'big'))
                    trim.append(list(f.read(3072)))
                    if len(trlab) == batch_size:
                        trlab = np.array(trlab)
                        trim = np.array(trim)
                        trim.shape = (batch_size, 3, 32, 32)
                        trim = trim.transpose([0, 2, 3, 1])
                        trim = trim / 255
                        yield ({"images": trim}, trlab)
                        trim, trlab = [], []
                if len(trlab) > 0:
                    trlab = np.array(trlab)
                    trim = np.array(trim)
                    trim.shape = (len(trlab), 3, 32, 32)
                    trim = trim.transpose([0, 2, 3, 1])
                    trim = trim / 255
                    yield ({"images": trim}, trlab)


def cifar10_test():
    test_batch = os.path.join(cifar10_dir, "test_batch.bin")
    teim, telab = [], []

    with open(test_batch, "rb") as f:
        for _ in range(10000):
            telab.append(int.from_bytes(f.read(1), 'big'))
            teim.append(list(f.read(3072)))
            if len(telab) == batch_size:
                telab = np.array(telab)
                teim = np.array(teim)
                teim.shape = (batch_size, 3, 32, 32)
                teim = teim.transpose([0, 2, 3, 1])
                teim = teim / 255
                yield ({"images": teim}, telab)
                teim, telab = [], []
        if len(telab) > 0:
            telab = np.array(telab)
            teim = np.array(teim)
            teim.shape = (len(telab), 3, 32, 32)
            teim = teim.transpose([0, 2, 3, 1])
            teim = teim / 255
            yield ({"images": teim}, telab)


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
