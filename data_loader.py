from scipy.io import loadmat
import pandas as pd
import glob
import os
from tqdm import tqdm


EXPECTED_SAMPLE_RATE = 400
DATA_LOCATION = os.path.join(os.path.dirname(__file__), 'input')


def sort_data_by_key(fn):
    """We need to group similar hours together.

    First sort by target and then by index.
    """
    parts = os.path.basename(fn)[:-4].split('_')
    if len(parts) == 2:
        return int(parts[-1])
    else:
        _patient, index, target = parts
        return (int(target), int(index))


def data_files(prefix='train', patient='*'):
    files = glob.glob(os.path.join(DATA_LOCATION,
                                   '%s_%s' % (prefix, patient),
                                   '*.mat'))
    files.sort(key=sort_data_by_key)
    return files


def load_metadata(max_results=99999999, patient='*', prefix='train'):
    files = data_files(prefix=prefix, patient=patient)
    max_results = min(max_results, len(files))
    for filename in tqdm(files[:max_results]):
        yield extract_path(filename), filename


def extract_path(filename):
    id_str = os.path.basename(filename)[:-4]
    arr = id_str.split("_")
    label = {
        'patient': int(arr[0]),
        'id': int(arr[1]),
    }
    if len(arr) > 2:
        label['result'] = int(arr[2])
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

    sample_rate = mat['dataStruct']['iEEGsamplingRate']
    assert sample_rate == EXPECTED_SAMPLE_RATE, 'Got a sample rate of %s' % sample_rate

    df = pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])
    return df, sequence
