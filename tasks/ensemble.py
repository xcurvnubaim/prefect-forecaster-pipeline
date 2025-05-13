import numpy as np
from typing import Any

class EMEnsembleRegressor:
    def __init__(self, models: dict[str, Any]):
        self.models = models
        self.weights = np.ones(len(models)) / len(models)
        self.model_names = list(models.keys())

    def fit(self, X, y, max_iter=100, tol=1e-4):
        preds = np.array([self.models[name].predict(X) for name in self.model_names])
        n_models, n_samples = preds.shape
        weights = self.weights.copy()

        for _ in range(max_iter):
            # E-step: compute responsibilities
            errors = np.array([np.square(y - preds[i]) for i in range(n_models)])
            responsibilities = np.exp(-errors)
            responsibilities /= responsibilities.sum(axis=0)

            # M-step: update weights
            new_weights = responsibilities.mean(axis=1)
            new_weights /= new_weights.sum()

            if np.linalg.norm(new_weights - weights) < tol:
                break
            weights = new_weights

        self.weights = weights

    def predict(self, X):
        preds = np.array([self.models[name].predict(X) for name in self.model_names])
        return np.average(preds, axis=0, weights=self.weights)
