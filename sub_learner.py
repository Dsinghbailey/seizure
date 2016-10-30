import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, \
    cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.random_projection import GaussianRandomProjection

import math
import datetime
import random
import os
import pprint

from feature_gen import create_train, results, mem
from data_loader import load_metadata


CLS_GBM = Pipeline([
    # ('transform', GaussianRandomProjection(random_state=2016)),
    ('classifier', GradientBoostingClassifier(n_estimators=100,
                                              random_state=2016))])
PARAMS_GBM = {
    'classifier__subsample': [0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    'classifier__min_samples_leaf': [1, 2, 3, 4],
    'classifier__min_samples_split': [int(2 * math.sqrt(2) ** i)
                                      for i in range(0, 8)],
    'classifier__max_leaf_nodes': [int(4 * 1.25**i) for i in range(1, 10)],
    'classifier__min_impurity_split': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
    'classifier__max_features': ['log2', 'sqrt', 0.5, 0.75, 1.0],
    'classifier__learning_rate': [0.1 * 1.05**i for i in range(-5, 5)],
    # 'transform__n_components': [60, 80, 120, 160, 240]
}

CLS_LOG_REG = Pipeline([
    ('normalize', Normalizer()),
    ('clf', LogisticRegression())])

PARAMS_LOG_REG = {
    'clf__C': [0.0125, 0.025, 0.05, 0.1, 0.2, .4, .8, 1.2, 2.5, 5, 10, 20, 40, 80, 160]
}


def find_hyperparameters(patient, X, y, cls, params):
    clf = RandomizedSearchCV(
        cls,
        params,
        verbose=3,
        n_iter=5,
        iid=False,
        cv=list(cv(patient, X, y, count=3, random_seed=0)),
        scoring='roc_auc',
        n_jobs=3
    )
    clf.fit(X, y)
    print "best params:"
    pprint.pprint(clf.best_params_)
    return clf.best_estimator_


def cv_score(patient, clf, X, y):
    scores = cross_val_score(clf, X, y,
                             verbose=2,
                             cv=list(cv(patient, X, y, count=10,
                                        random_seed=2016)),
                             scoring='roc_auc', n_jobs=3)
    print scores
    print scores.mean()


def file_hash(filename):
    parts = filename[:-4].split('_')
    outcome = 0
    if len(parts) == 3:
        patient, hour, outcome = parts
    else:
        patient, hour = parts
    return ((int(hour)-1) / 6) * 1000 + int(patient) * 10 + int(outcome)


def cv(patient, X, y, count=3, test_size=0.3, random_seed=None):
    hour_list = map(file_hash, filenames(patient, 'train'))
    multiple = int(X.shape[0])/len(hour_list)
    hour_list = list(hour for hour in hour_list
                          for _ in range(multiple))
    return GroupShuffleSplit(count, test_size=test_size,
                             random_state=random_seed
    ).split(X, y, groups=hour_list)


@mem.cache
def sub_learn(patient=1):
    X, y = create_train(patient=patient)
    clf = find_hyperparameters(patient, X, y, CLS_GBM, PARAMS_GBM)
    cv_score(patient, clf, X, y)
    clf.fit(X, y)

    X_meta = upper_features(clf, X, patient)
    return clf, X_meta, y_hours(patient)


def filenames(patient, prefix):
    return [os.path.basename(fn) for (_, fn) in
            load_metadata(patient=patient,
                          max_results=999999,
                          prefix=prefix)]


def upper_features(clf, X, patient, prefix='train'):
    hour_list = map(file_hash, filenames(patient, prefix))
    hour_list = list(set(hour_list))
    segments_per_hour = int(X.shape[0])/len(hour_list)
    # each row is an hour, with a feature per segment.
    featuress = []
    for hour in range(len(hour_list)):
        start = hour*segments_per_hour
        features = clf.predict_proba(
            X[start:(start + segments_per_hour), :])[:, 1]
        featuress.append(np.transpose(features))
    return np.vstack(featuress)


def y_hours(patient):
    y_10_min = np.array([
        results(file_info)
        for file_info, _ in load_metadata(max_results=99999,
                                          prefix='train',
                                          patient=patient)])
    return y_10_min[0::6]
