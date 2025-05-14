import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from typing import Any

class BMAEnsembleRegressor:
    """
    Bayesian Model Averaging ensemble regressor using inverse-MSE weighting with K-Fold cross-validation.

    Parameters
    ----------
    models : dict[str, object]
        Dictionary of base models. Keys are model names and values are model objects with `fit` and `predict` methods.
    n_splits : int, default=5
        Number of folds for K-Fold cross-validation.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    def __init__(self, models: dict[str, object], n_splits: int = 5, random_state: int = 42):
        self.models = models
        self.n_splits = n_splits
        self.random_state = random_state
        self.mse_scores = {}
        self.weights_ = None

    def fit(self, X: np.ndarray | Any, y: np.ndarray | Any):
        """
        Fit the BMA ensemble by computing weights from inverse MSE using K-Fold CV.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X_df = X.copy() if not isinstance(X, np.ndarray) else X
        y = np.asarray(y).ravel()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for name, model in self.models.items():
            cv_errors = []
            for train_idx, val_idx in kf.split(X_df):
                X_train = X_df.iloc[train_idx] if hasattr(X_df, "iloc") else X_df[train_idx]
                X_val = X_df.iloc[val_idx] if hasattr(X_df, "iloc") else X_df[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                try:
                    model.fit(X_train, y_train)
                    y_pred = np.asarray(model.predict(X_val)).ravel()
                    mse = mean_squared_error(y_val, y_pred)
                    cv_errors.append(mse)
                except Exception as e:
                    print(f"Model {name} failed during CV fold: {e}")

            if cv_errors:
                avg_mse = np.mean(cv_errors)
                self.mse_scores[name] = max(avg_mse, 1e-9)
            else:
                print(f"Model {name} failed all CV folds.")

        valid_mse = {name: mse for name, mse in self.mse_scores.items() if mse > 0 and name in self.models}
        if not valid_mse:
            raise ValueError("No valid models with positive MSE scores found.")

        inverse_sum = sum(1.0 / mse for mse in valid_mse.values())
        self.weights_ = {
            name: (1.0 / mse) / inverse_sum
            for name, mse in valid_mse.items()
        }
        return self

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        """
        Predict using the BMA ensemble.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Weighted average of predictions.
        """
        if self.weights_ is None:
            raise RuntimeError("Model must be fit before prediction.")

        weighted_preds = np.zeros(len(X))
        for name, model in self.models.items():
            if name in self.weights_:
                pred = np.asarray(model.predict(X)).ravel()
                weighted_preds += pred * self.weights_[name]

        return weighted_preds
