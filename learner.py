"""Learn on features derived from the vector of sub-learners probabilities

x := 8-second sub-learner vector probability vector for 1 hour
y := ground truth target

there is a 5 minute gap between the last sub-learner and the seizure event.

minimize avg {
  let p = a + b * avg(x) + c * linear(x) + d * exp(x) + e * quadratic^2(x) + f * max(x) + g * var(x)
  in -y * ln(p) - (1-y) * ln(1 - p)

} where a, b in Real

where linear(x) = sum_{i=0}^{n-1} {x[i] * 1/(n-i + 5 * 60 / 8)}
      etc
"""
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
import numpy as np


def learn(X, y):
    clf = Pipeline([
        ('features', AverageLearner()),
        ('clf', LogisticRegression())
    ])

    clf.fit(X, y)
    return clf


def AverageLearner():
    return FunctionTransformer(
        lambda X, y=None: X.mean(axis=1)[:, np.newaxis])


def weight_function(f):
    def func(X, y=None):
        return np.dot(X, f(X))
    return func

def WeightedAverage(f):
    return FunctionTransformer(weight_function(f))


BLOCK_DURATION = 8
GAP_DURATION = 5 * 60


def linear_weights(X):
    n = X.shape[1]
    return np.array([1/(n-i + GAP_DURATION / BLOCK_DURATION)])
