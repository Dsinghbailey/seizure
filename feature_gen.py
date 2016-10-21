from data_loader import load_data
import numpy as np
from joblib import Memory
mem = Memory(cachedir='tmp/joblib')


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
    return file_info['result']
