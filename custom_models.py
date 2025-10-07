import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class EnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for name, model in self.models.items():
            model.fit(X, y)
        return self

    def predict(self, X):
        preds = np.column_stack([
            model.predict(X) for model in self.models.values()
        ])
        return preds.mean(axis=1)

