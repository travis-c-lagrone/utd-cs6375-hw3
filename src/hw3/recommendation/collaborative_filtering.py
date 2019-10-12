"""Collaborative filtering algorithms."""

from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np


class CorrelationCollaborativeFilter(BaseEstimator, ClassifierMixin):
    """Correlation collaborative filtering classifier."""

    X_: ndarray
    _predictions: ndarray

    def fit(self, X: ndarray) -> 'CorrelationCollaborativeFilter':
        """Fit this correlation collaborative filtering classifier with data.

        Args:
            X (ndarray): Votes (values) by users (rows) on items (columns).
                Votes are positive real numbers, where zero indicates no vote.

        Returns:
            CorrelationCollaborativeFilter: This classifier, fit with ``X``.

        """
        encapsulated = np.copy(X)
        softened = np.nan_to_num(encapsulated, copy=False)
        self.X_ = X = softened

        n = X.shape[0]

        means = X.mean(axis=1)  # [user]
        marginals = np.empty(X.shape)  # [user, item]
        for u in range(n):
            marginals[u, :] = X[u] - means[u]

        exists = X.astype(bool)  # [user, item]
        weights = np.empty((n, n))  # [user, user]
        for u1 in range(n):
            for u2 in range(u1, n):
                if u1 == u2:
                    weights[u1, u2] = 0
                    continue
                boths = exists[u1] & exists[u2]  # [item]
                u1_marginals = marginals[u1][boths]
                u2_marginals = marginals[u2][boths]
                u1_dot_u1 = np.dot(u1_marginals, u1_marginals)
                u1_dot_u2 = np.dot(u1_marginals, u2_marginals)
                u2_dot_u2 = np.dot(u2_marginals, u2_marginals)
                corr = u1_dot_u2 / np.sqrt(u1_dot_u1 * u2_dot_u2)
                weights[u1, u2] = corr

        normalizers = np.empty(n)  # [user]
        for u1 in range(n):
            sliced = (weights[min(u1, u2), max(u1, u2)] for u2 in range(n))
            normed = (np.abs(w) for w in sliced)
            summed = sum(normed)
            inverted = 1 / summed
            normalizers[u1] = inverted

        self._predictions = np.empty(X.shape)  # [user, item]
        for u1 in range(n):
            indexed = (u2 for u2 in range(n) if u2 != u1)
            weighted = [marginals[u2] * weights[u1, u2] for u2 in indexed]
            summed = np.sum(weighted, axis=0)
            normalized = summed * normalizers[u1]
            offset = normalized + means[u1]
            self._predictions[u1, :] = offset

        return self

    def predict(self, y: int) -> ndarray:
        """Predict the votes by user ``y`` on all items.

        Args:
            y (int): The user (row index) for which to predict votes on all items.

        Returns:
            ndarray: The predicted votes by user ``y`` on all items.
                A 1d array of nonnegative real numbers, where zero indicates no prediction.

        """
        return self._predictions[y, :].copy()
