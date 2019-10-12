"""Collaborative filtering algorithms."""

from typing import Any

from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin


class CorrelationCollaborativeFilter(BaseEstimator, ClassifierMixin):
    """Correlation collaborative filtering classifier."""

    def fit(self, X: ndarray) -> 'CorrelationCollaborativeFilter':
        """Fit this correlation collaborative filtering classifier with data.

        Args:
            X (ndarray): Votes (values) by users (rows) on items (columns).
                Votes are nullable real numbers.

        Returns:
            CorrelationCollaborativeFilter: This classifier, fit with ``X``.

        """
        self.X_ = X
        return self

    def predict(self, y: int) -> ndarray:
        """Predict the votes by user ``y`` on all items.

        Args:
            y (int): The user (row index) for which to predict votes on all items.

        Returns:
            ndarray: The predicted votes by user ``y`` on all items.
                A 1d array of nullable real numbers.

        """
        pass  # TODO Implement CorrelationCollaborativeFilter.predict(self, y)
