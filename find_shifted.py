import argparse
import sys

import numpy as np
from tqdm import tqdm

from feature_gen import mem
from data_loader import load_matlab_file, load_metadata


def signature(X, threshold=10):
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


def load_hour(hour, patient, prefix='test', channel=7, down_sample=1):
    start = hour * 6 + 1
    parts = []
    for i in range(start, start+6):
        fn = r'input\%s_%s\%s_%s' % (prefix, patient, patient, i)
        data, _ = load_matlab_file(fn)
        parts.append(data.as_matrix()[:, channel])

    return gaps(np.hstack(parts)[::down_sample])


@mem.cache
def load_hours(patient, prefix):
    files = [result[1]
             for result in load_metadata(patient=patient, prefix=prefix)]

    print 'Loading files...'
    return [
        load_hour(hour, patient, prefix)
        for hour in tqdm(range(len(files) / 6))]


def find_all_overlaps(patient, prefix):
    hours = load_hours(patient, prefix)

    hours_sigs = map(signature, hours)

    print 'Finding overlaps...'
    for i in tqdm(range(len(hours))):
        for j in range(i):
            overlap = find_overlap(hours[i], hours[j])
            if overlap:
                yield i, j, overlap


def find_overlap(a, b):
    return contained_in(a, b) or contained_in(b, a)


def contained_in(a, b):
    if all(a[i] == 0 for i in range(10)):
        i = a.nonzero()[0][0]
    else:
        i = 0
    for j in range(len(b)):
        if match_at(a, b, i, j):
            return j - i
    return None


MAX_MATCH_CHECK = 10


def match_at(a, b, i, j):
    count = 0
    while  j < len(b) and a[i] == b[j] and count <= MAX_MATCH_CHECK:
        i += 1
        j += 1
    return (j == len(b)) or a[i] == b[j]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient', default=1, type=int)
    parser.add_argument('--prefix', default='test')
    args = parser.parse_args(sys.argv)

    for i, j, overlap in find_all_overlaps(args.patient, args.prefix):
        print '%s-%s with overlap %s' % (i, j, overlap)


if __name__ == '__main__':
    main()
