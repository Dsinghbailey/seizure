from sub_learner import sub_learn, filenames, upper_features
from feature_gen import create_train
#Todo:write learner
from learner import learn
import datetime
import itertools as it
DESTINATION = 'submission_{:%Y-%m-%d_%H-%M-%S}.out'.format(datetime.datetime.now())

def make_submission(clfs, fp, patient):
      X_test, _ = create_train(patient=patient, prefix='test')
      X_test = upper_features(clfs[patient], X_test, patient, prefix='test') 
      y_hours = clfs[patient].predict_proba(X_test)[:, 1]
      # are hours grouped together? if not the following line is a mistake
      y_submission = list(it.repeat(y, 6) for y in y_hours)
      for filename, proba in zip(filenames(patient, 'test'),
                                 y_submission):
          fp.write('%s,%s\n' % (filename, proba))


if __name__ == '__main__':
    with open(DESTINATION, 'w') as fp:
        for patient in [1, 2, 3]:
            X, y = sub_learn(patient=patient)
            clfs = []
            clfs[patient] = learn(X, y)
            fp.write('File,Class\n')
            make_submission(clfs, fp, patient)
