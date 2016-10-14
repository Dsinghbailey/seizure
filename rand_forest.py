from sklearn.ensemble import RandomForestClassifier
from feature_gen import create_train
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import ShuffleSplit, cross_val_score
import numpy as np


def forest_learn():
    clf = RandomForestClassifier()
    X, y = create_train()
    cv = ShuffleSplit(np.shape(X)[0], n_iter=3, test_size=0.3, random_state=0)
    print cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
