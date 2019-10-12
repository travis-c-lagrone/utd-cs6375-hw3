"""Correlation collaborative filtering for recommendation."""

from numpy import ndarray
from sklearn.base import BaseEstimator

import numpy as np


class CorrelationCollaborativeFilter(BaseEstimator):
    """Correlation collaborative filtering recommender."""

    X_: ndarray
    _predictions: ndarray

    def fit(self, X: ndarray) -> 'CorrelationCollaborativeFilter':
        r"""Fit this correlation collaborative filtering recommender with data.

        Eagerly computes all predictions. Eager prediction is at least as
        efficient than lazy execution, assuming that the count of distinct
        predictions that will be made is :math:`\Omega(\sqrt{|X|})`.

        Args:
            X (ndarray): Votes (values) by users (rows) on items (columns).
                Votes are positive real numbers, where zero indicates no vote.

        Returns:
            CorrelationCollaborativeFilter: This recommender, fit with ``X``.

        """
        encapsulated = np.copy(X)
        softened = np.nan_to_num(encapsulated, copy=False)
        self.X_ = X = softened

        n = X.shape[0]

        means = X.mean(axis=1)  # [user]
        marginals = np.stack(X[u] - means[u] for u in range(n))  # [user, item]

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

            coords = ((min(u1, u2), max(u1, u2)) for u2 in range(n))
            multi_index = list(zip(*coords))
            flat_indexes = np.ravel_multi_index(multi_index, weights.shape)
            sliced = weights.ravel()[flat_indexes]

            normed = (np.abs(w) for w in sliced)  # L1 norm
            summed = sum(normed)
            inverted = 1 / summed

            normalizers[u1] = inverted

        self._predictions = np.empty(X.shape)  # [user, item]
        for u1 in range(n):

            others = (u2 for u2 in range(n) if u2 != u1)  # all other users

            weighted = [marginals[u2] * weights[u1, u2] for u2 in others]
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
