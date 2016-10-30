import numpy as np

from data_loader import load_matlab_file


def signature(X, threshold):
    fourtytwos = (X < threshold + 1.0) == (X >= threshold)
    nonzeros = fourtytwos.nonzero()
    if len(nonzeros) > 1:
        all_times, channels = nonzeros
        nonzeros = all_times[channels == 7]
    else:
        nonzeros = nonzeros[0]
    return gaps(nonzeros)


def gaps(X):
    return X[1:] - X[:-1]


def load_file(number, patient):
    fn = r'input\test_%s\%s_%s' % (patient, patient, number)
    data, _ = load_matlab_file(fn)
    return gaps(data.as_matrix()[:, 7])
