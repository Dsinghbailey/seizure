import pandas as pd
from data_loader import load_data
import numpy as np


def create_train():
    gen = load_data()
    train = pd.DataFrame()
    for i, (file_info, (data, sequence)) in gen:
        debug_on_sequence_weirdness(i, sequence)
        train.append(list(fft(data)))


def debug_on_sequence_weirdness(i, sequence):
    if i % 6 != sequence % 6:
        import pdb; pdb.set_trace()


def fft(data):
    for i in range(data.shape()[1]):
        for col in data[:, i]:
            yield np.absolute(np.fft.rfft(col))
