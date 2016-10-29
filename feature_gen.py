import os

import numpy as np
from sklearn.preprocessing import normalize
from joblib import Memory, Parallel, delayed
# from joblib.pool import has_shareable_memory
from multiprocessing import Pool

from data_loader import load_metadata, load_matlab_file


mem = Memory(cachedir=os.path.join('tmp', 'joblib'))
FREQ = 400
SIZE = 8 * FREQ
OVERLAP = .5


@mem.cache
def create_train(prefix='train', patient='*', n=99999999):
    """Load data from disk and return features and targets."""
    gen = load_metadata(max_results=n, prefix=prefix, patient=patient)

    data_setss = Parallel(n_jobs=4, max_nbytes=1e6)(
        delayed(features_from_mat)(file_info, filename)
        for file_info, filename in gen
    )

    xss = []
    yss = []
    for block_set in zip(*data_setss):
        xs, ys = zip(*block_set)
        xss.append(xs)
        yss.extend(ys)

    X = np.vstack(xss)
    return X, yss


def precompute_features():
    create_train(prefix='train')
    create_train(prefix='test')


def postprocess_features(X):
    """Normalizes and re-buckets the features."""
    # Xlog = np.log(X + 0.0000000001)
    # X = np.hstack([Xlog, X])
    X = normalize(X, axis=0, copy=False)
    return X

def bands(X, n_bands=10):
    n = X.shape[0]
    band_size = n / n_bands
    X = np.hstack(
        [[X[0]]] +
        [X[i*band_size+1:i*band_size+1+band_size].mean()
         for i in range(10)])
    return X


def features_from_mat(file_info, filename, stride=20):
    """Loads a matlab file, and yields the smoothed FFT in 8-second blocks."""
    data, _sequence = load_matlab_file(filename)
    blocks = []

    def block_at(i):
        start = int(SIZE * i * OVERLAP)
        fin = start + SIZE
        return data.iloc[start:fin, :]

    def features(block):
        X = np.concatenate(list(fft(block, stride=stride)))
        y = results(file_info)
        return bands(X), y

    i_list = range(data.shape[0]//int(SIZE * OVERLAP) - 1)
    blocks = map(block_at, i_list)

    return map(features, blocks)


# The fft measures up to 200hz (which is sample frequency / 2), but we only
# care about frequencies up to 1..25hz
def trim_fft(fft):
    """Trims unneccessary frequencies from the fft."""
    avg = fft[0]
    one_hz = fft.shape[0] // (FREQ / 2)
    return np.hstack([[avg],
                      fft[one_hz:len(fft) // 8]
    ])

# we have 200 * 10 * 60 samples
# we have 200 * 10 * 30 frequencies in the fft
# frequencies range from 0 hz to 200 hz
# so fft element at index i has frequency i / (10 * 40)
def smooth_fft(fft, stride):
    """Average adjancent frequencies proportional to the frequency."""
    yield fft[0]
    i = 1
    while i < len(fft):
        bucket_size_elems = max(5, i / stride)
        new_i = i + bucket_size_elems
        yield fft[i:min(len(fft), new_i)].mean()
        i = new_i


def regularize_fft(fft, stride):
    fft_magnitude = np.absolute(fft)
    return np.array(list(smooth_fft(
        trim_fft(fft_magnitude),
        stride=stride)))


def fft(data, stride):
    for i in range(data.shape[1]):
        yield regularize_fft(
            np.fft.rfft(data.iloc[:, i]),
            stride=stride)


def results(file_info):
    return file_info.get('result')


