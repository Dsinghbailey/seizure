import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection

import math
import datetime
import random
import os
import pprint

from feature_gen import create_train, precompute_features
from data_loader import load_metadata


CLS_GBM = Pipeline([
    # ('transform', GaussianRandomProjection(random_state=2016)),
    ('classifier', GradientBoostingClassifier(n_estimators=200,
                                              random_state=2016))])
PARAMS_GBM = {
    'classifier__subsample': [0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    'classifier__min_samples_leaf': [1, 2, 3, 4],
    'classifier__min_samples_split': [int(2 * math.sqrt(2) ** i)
                                      for i in range(0, 8)],
    'classifier__max_leaf_nodes': [int(4 * 1.25**i) for i in range(1, 10)],
    'classifier__min_impurity_split': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
    'classifier__max_features': ['log2', 'sqrt', 0.5, 0.75, 1.0],
    'classifier__learning_rate': [0.1/2 * 1.05**i for i in range(-5, 5)],
    # 'transform__n_components': [60, 80, 120, 160, 240]
}

CLS_LOG_REG = LogisticRegression()
PARAMS_LOG_REG = {
    'C': [0.0125, 0.025, 0.05, 0.1, 0.2, .4, .8, 1.2, 2.5, 5, 10, 20, 40, 80, 160]
}


def find_hyperparameters(patient, X, y, cls, params):
    clf = RandomizedSearchCV(
        cls,
        params,
        verbose=3,
        n_iter=10,
        iid=False,
        cv=list(cv(patient, X, y, count=5, random_seed=0)),
        scoring='roc_auc',
        n_jobs=4
    )
    clf.fit(X, y)
    print "best params:"
    pprint.pprint(clf.best_params_)
    return clf.best_estimator_


def cv_score(patient, clf, X, y):
    scores = cross_val_score(clf, X, y,
                             cv=list(cv(patient, X, y, count=20,
                                        random_seed=2016)),
                             scoring='roc_auc', n_jobs=3)
    print scores
    print scores.mean()


def patient_hour(filename):
    patient, hour, _ = filename.split('_')
    return (int(hour) / 6) * 1000 + int(patient)


def cv(patient, X, y, count=3, test_size=0.3, random_seed=None):
    hour_list = map(patient_hour, filenames(patient, 'train'))

    return GroupShuffleSplit(count, test_size=test_size,
                             random_state=random_seed
    ).split(X, y, groups=hour_list)


def log_learn():
    # precompute_features()
    clfs = {}
    for patient in (1, 2, 3):
      X, y = create_train(patient=patient)
      print X.shape
      clf = find_hyperparameters(patient, X, y, CLS_GBM, PARAMS_GBM)
      cv_score(patient, clf, X, y)
      clf.fit(X, y)
      clfs[patient] = clf

    with open(DESTINATION, 'w') as fp:
      fp.write('File,Class\n')
      make_submission(clfs, fp)


DESTINATION = 'submission_{:%Y-%m-%d_%H-%M-%S}.out'.format(datetime.datetime.now())


def filenames(patient, prefix):
    return [os.path.basename(fn) for (_, fn) in
            load_metadata(patient=patient,
                          max_results=999999,
                          prefix=prefix)]


def make_submission(clfs, fp):
    for patient in [1,2,3]:
      X_test, _ = create_train(patient=patient, prefix='test')
      y_submission = clfs[patient].predict_proba(X_test)[:,1]

      for filename, proba in zip(filenames(patient, 'test'),
                                 y_submission):
          fp.write('%s,%s\n' % (filename, proba))
