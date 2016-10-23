import os

import numpy as np
from sklearn.preprocessing import normalize
from joblib import Memory

from data_loader import load_metadata, load_matlab_file


mem = Memory(cachedir=os.path.join('tmp', 'joblib'))


def create_train(n=99999999, prefix='train'):
    gen = load_metadata(n, prefix=prefix)
    xs = []
    ys = []
    for file_info, filename in gen:
        X, y = features_from_mat(file_info, filename)
        xs.append(X)
        ys.append(y)

    xs = postprocess_features(np.vstack(xs))
    return xs, ys


def precompute_features():
    create_train(prefix='train')
    create_train(prefix='test')


def postprocess_features(X):
    # Xlog = np.log(X + 0.0000000001)
    # X = np.hstack([Xlog, X])
    X = normalize(X, axis=0, copy=False)
    return X


@mem.cache
def features_from_mat(file_info, filename):
    data, _sequence = load_matlab_file(filename)
    X = np.concatenate(list(fft(data)))
    y = results(file_info)
    return X, y


# The fft measures up to 200hz, but we only care about frequencies up to 50hz
def trim_fft(fft):
    return fft[:len(fft) // 3]

# we have 150 * 10 * 60 samples
# we have 150 * 10 * 30 frequencies in the fft
# frequencies range from 0 hz to 150 hz
# so fft element at index i has frequency i / (10 * 30)

def smooth_fft(fft, stride=10):
    # Average adjancent frequencies proportional to the frequency.
    i = 0
    while i < len(fft):
        bucket_size_elems = max(1, i / stride)
        new_i = i + bucket_size_elems
        yield fft[i:min(len(fft), new_i)].mean()
        i = new_i


def regularize_fft(fft):
    fft_magnitude = np.absolute(fft)
    return np.array(list(smooth_fft(trim_fft(fft_magnitude))))


def fft(data):
    for i in range(data.shape[1]):
        yield regularize_fft(
            np.fft.rfft(data.iloc[:, i]))


def results(file_info):
    return file_info.get('result')


