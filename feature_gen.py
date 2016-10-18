from data_loader import load_data
import numpy as np
from joblib import Memory
mem = Memory(cachedir='/tmp/joblib')


@mem.cache
def create_train():
    gen = load_data(600)
    X = []
    y = []
    for file_info, (data, sequence) in gen:
        X.append(np.concatenate(list(fft(data))))
        y.append(results(file_info))

    return (X, y)


def debug_on_sequence_weirdness(i, sequence):
    if i % 6 != sequence % 6:
        import pdb
        pdb.set_trace()


def fft(data):
    for i in range(data.shape[1]):
        yield np.absolute(np.fft.rfft(data.iloc[:, i]))


def results(file_info):
    return file_info['result']
