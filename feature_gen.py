import os

import numpy as np
from sklearn.preprocessing import normalize
from joblib import Memory

from data_loader import load_metadata, load_matlab_file


mem = Memory(cachedir=os.path.join('tmp', 'joblib'))


def create_train(prefix='train', patient='*', n=99999999):
    gen = load_metadata(max_results=n, prefix=prefix, patient=patient)
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
    n = X.shape[1]
    band_size = n / 10
    X = np.vstack([
        np.hstack([[X[j, 0]]] +
                  [X[j, i*band_size+1:i*band_size+1+band_size] for i in range(10)]
        )
        for j in range(X.shape[0])
    ])
    # X_smooth = np.vstack([
    #     rebucket(X[i, :], stride=3)
    #     for i in range(X.shape[0])
    # ])

    return X


def rebucket(X, stride):
    n_col = 16
    n = X.shape[0]
    indices = range(0, n, n / n_col)
    buckets = []
    for old_index, new_index in zip(indices, indices[1:] + [-1]):
        bucket = X[old_index:new_index]
        buckets.append(list(smooth_fft(bucket, stride)))

    return np.concatenate(buckets)


@mem.cache
def features_from_mat(file_info, filename, stride=20):
    data, _sequence = load_matlab_file(filename)
    X = np.concatenate(list(fft(data, stride=stride)))
    y = results(file_info)
    return X, y


# The fft measures up to 200hz, but we only care about frequencies up to 50hz
def trim_fft(fft):
    return fft[:len(fft) // 3]

# we have 150 * 10 * 60 samples
# we have 150 * 10 * 30 frequencies in the fft
# frequencies range from 0 hz to 150 hz
# so fft element at index i has frequency i / (10 * 30)

def smooth_fft(fft, stride):
    # Average adjancent frequencies proportional to the frequency.
    i = 0
    while i < len(fft):
        bucket_size_elems = max(1, i / stride)
        new_i = i + bucket_size_elems
        yield fft[i:min(len(fft), new_i)].mean()
        i = new_i


def regularize_fft(fft, stride):
    fft_magnitude = np.absolute(fft)
    return np.array(list(smooth_fft(
        trim_fft(fft_magnitude),
        stride=stride)))


def fft(data, stride=10):
    for i in range(data.shape[1]):
        yield regularize_fft(
            np.fft.rfft(data.iloc[:, i]),
            stride=stride)


def results(file_info):
    return file_info.get('result')


