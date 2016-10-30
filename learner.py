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
    # clf = Pipeline([
    #     ('features', FeatureUnion([
    #         ('Average', AverageLearner()),
    #         # ('Linear', WeightedAverage(linear_weights)),
    #         ('Percentiles', Percentiles())])),
    #     ('clf', LogisticRegression(C=50))
    # ])
    clf = AverageLearner()

    clf.fit(X, y)
    return clf


def Percentiles(buckets=10):

    def ExtractPercentiles(X, y=None):
        out = []
        for row in range(X.shape[0]):
            row_sorted = np.sort(X[row, :])
            bucket_size = row_sorted.shape[0] / buckets
            row_out = []
            for bucket in range(buckets + 1):
                index = min(bucket * bucket_size, row_sorted.shape[0] - 1)
                row_out.append(row_sorted[index])
            out.append(row_out)
        return np.array(out)

    return FunctionTransformer(ExtractPercentiles)


class AverageLearner(object):
    def fit(self,aX, y):
        pass

    def predict_proba(self, X):
        avg = average(X)
        return np.hstack([1-avg, avg])

    def transform(self, X):
        return average(X)


def average(X, y=None):
    return X.mean(axis=1)[:, np.newaxis]


def AverageTransformer():
    return FunctionTransformer(average)


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
