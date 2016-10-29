import datetime
import itertools as it
import pickle

from feature_gen import create_train
from learner import learn
from sub_learner import sub_learn, filenames, upper_features

DESTINATION = 'submission_{:%Y-%m-%d_%H-%M-%S}'.format(
      datetime.datetime.now())


def make_submission(clfs, fp, patient):
      X_test, _ = create_train(patient=patient, prefix='test')
      pre_clf, post_clf = clfs[patient]
      X_test = upper_features(pre_clf, X_test, patient, prefix='test')
      y_hours = post_clf.predict_proba(X_test)[:, 1]

      # All 6 ten-min segments get the same prediction
      y_submission = [y for y in y_hours
                        for _ in range(6)]
      for filename, proba in zip(filenames(patient, 'test'),
                                 y_submission):
          fp.write('%s,%s\n' % (filename, proba))


def main(clfs=None):
    with open(DESTINATION + '.out', 'w') as fp:
        clfs = clfs or {}
        fp.write('File,Class\n')
        for patient in [1, 2, 3]:
            clf, X, y = sub_learn(patient=patient)
            if patient not in clfs:
                clfs[patient] = clf, learn(X, y)
            make_submission(clfs, fp, patient)

    # with open(DESTINATION + '.pickle', 'w') as fp:
    #       pickle.dump(clfs, fp)


if __name__ == '__main__':
      main()
