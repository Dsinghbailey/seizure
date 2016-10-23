from sklearn.linear_model import LogisticRegression
from feature_gen import create_train, precompute_features
from data_loader import load_metadata
from sklearn.cross_validation import ShuffleSplit, cross_val_score
import numpy as np
from sklearn.grid_search import GridSearchCV

import datetime
import os
import pprint


def find_hyperparameters(X, y):
    cv = ShuffleSplit(np.shape(X)[0], n_iter=2, test_size=0.1, random_state=0)
    clf = GridSearchCV(
        LogisticRegression(),
        {'C': [1.3 ** i for i in range(15, 25, 1)]},
        verbose=3,
        cv=cv,
        scoring='roc_auc',
        n_jobs=3
    )
    clf.fit(X, y)
    print "GridSearchCV: "
    pprint.pprint(clf.__dict__)
    return clf.best_estimator_


def cv_score(clf, X, y):
    cv = ShuffleSplit(np.shape(X)[0], n_iter=10, test_size=0.1, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=3)
    print scores
    print scores.mean()


def log_learn():
    # precompute_features()
    X, y = create_train()
    print X.shape
    clf = find_hyperparameters(X, y)
    cv_score(clf, X, y)

    clf.fit(X, y)
    make_submission(clf)


DESTINATION = 'submission_{:%Y-%m-%d_%H-%M-%S}.out'.format(datetime.datetime.now()) 

def make_submission(clf):
    X_test, _ = create_train(prefix='test')
    filenames = [fn for (_, fn) in
                 load_metadata(999999, prefix='test')]

    y_submission = clf.predict_proba(X_test)[:,1]

    with open(DESTINATION, 'w') as fp:
        fp.write('File,Class\n')
        for filename, proba in zip(filenames, y_submission):
            fp.write('%s,%s\n' % (os.path.basename(filename), proba))
