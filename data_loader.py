from scipy.io import loadmat
import glob
import os


DATA_LOCATION = os.path.join(os.path.dirname(__file__), 'input')


def data_files():
  return glob.glob(os.path.join(DATA_LOCATION, '*', '*.mat'))


def load_data():
  for filename in data_files():
    yield load_matlab_file(filename)


def load_matlab_file(filename):
  # TODO: maybe check that the sampling rate and # channels and samples per
  # matrix is consistent for all the data we have
  d = loadmat(filename)
  return {'data': d['dataStruct'][0,0], 'sampling_rate': d['dataStruct'][01]}
