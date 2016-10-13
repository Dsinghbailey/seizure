from scipy.io import loadmat
import pandas as pd
import glob
import os


DATA_LOCATION = os.path.join(os.path.dirname(__file__), 'input')


def data_files():
    return glob.glob(os.path.join(DATA_LOCATION, '*', '*.mat'))


def load_data():
    for filename in data_files():
        yield extract_path(filename), load_matlab_file(filename)


def extract_path(filename):
    id_str = os.path.basename(filename)[:-4]
    arr = id_str.split("_")
    label = {'patient': int(arr[0]),
             'id': int(arr[1]),
             'result': int(arr[2])}
    return label


def load_matlab_file(path):
    # mat = loadmat(filename)
    # names = mat['dataStruct'].dtype.names
    # ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    # return ndata
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    sequence = -1
    if 'sequence' in names:
        sequence = mat['dataStruct']['sequence']

    df = pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])
    return df, sequence
