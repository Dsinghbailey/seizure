from sklearn.linear_model import LogisticRegression
from feature_gen import create_train
from sklearn.cross_validation import ShuffleSplit, cross_val_score
import numpy as np
from sklearn.grid_search import GridSearchCV


def log_learn():
    parameters = {'C': [1.2 ** i for i in range(-10,10, 5)]}
    lr = LogisticRegression()
    clf = GridSearchCV(lr, parameters, verbose=3)
    X, y = create_train()
    cv = ShuffleSplit(np.shape(X)[0], n_iter=2, test_size=0.3, random_state=0)
    print cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    print clf.best_params_
