import numpy as np
from typing import Any, Dict, List
import pandas as pd # Import pandas for type checking
import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn") # Ignore warnings from sklearn

class BMAEnsembleRegressor:
    """
    Ensemble Regressor using Bayesian Model Averaging (BMA) principles
    with Expectation-Maximization (EM) for parameter estimation.

    The weights and variances of the individual models in the ensemble are
    estimated using the EM algorithm. The BMA prediction is a weighted
    average of the individual model predictions. This approach is inspired by
    the methodology for combining multiple models, such as those discussed in
    the paper "Bayesian model averaging by combining deep learning models to
    improve lake water level prediction."
    """
    def __init__(self, models: Dict[str, Any]):
        """
        Initialize the BMAEnsembleRegressor.

        Args:
            models (Dict[str, Any]): A dictionary of underlying regression models.
                                     Keys are model names (str) and values are
                                     fitted model objects with a .predict() method.
        """
        if not models:
            raise ValueError("Models dictionary cannot be empty.")
        self.models_dict = models
        self.model_names = list(models.keys())
        self.n_models = len(self.model_names)
        
        # Initialize weights uniformly
        self.weights_ = np.ones(self.n_models) / self.n_models
        # Placeholder for fitted model error variances (sigma_k^2)
        self.sigmas_sq_ = np.ones(self.n_models)

    def _gaussian_pdf(self, x: np.ndarray, mu: np.ndarray, sigma_sq: np.ndarray) -> np.ndarray:
        """
        Calculate Gaussian PDF values.
        Assumes sigma_sq is variance.
        x, mu, sigma_sq are broadcastable.
        """
        # Add a small epsilon to sigma_sq to prevent division by zero or issues with very small variances
        epsilon = 1e-9
        return (1.0 / np.sqrt(2 * np.pi * (sigma_sq + epsilon))) * np.exp(-np.square(x - mu) / (2 * (sigma_sq + epsilon)))

    def fit(self, X: Any, y: Any, max_iter: int = 100, tol: float = 1e-4, min_variance: float = 1e-6):
        """
        Fit the ensemble weights and model error variances using the EM algorithm.

        Args:
            X (Any): Input features (n_samples, n_features). Can be pandas DataFrame or NumPy array.
            y (Any): Target values (n_samples,). Can be pandas Series or NumPy array.
            max_iter (int): Maximum number of EM iterations.
            tol (float): Tolerance for convergence of weights.
            min_variance (float): Minimum allowed variance for numerical stability.
        """
        # Convert X and y to NumPy arrays if they are pandas objects
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_np = X.to_numpy()
        else:
            X_np = np.asarray(X)

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_np = y.to_numpy()
        else:
            y_np = np.asarray(y)

        n_samples = X_np.shape[0]
        if n_samples == 0:
            raise ValueError("Input X cannot be empty.")
        if y_np.shape[0] != n_samples:
             raise ValueError("X and y must have the same number of samples.")
        
        if y_np.ndim == 1:
            y_col = y_np[:, np.newaxis] # Ensure y is a column vector for broadcasting (n_samples, 1)
        elif y_np.ndim == 2 and y_np.shape[1] == 1:
            y_col = y_np
        else:
            raise ValueError(f"y has an unexpected shape: {y_np.shape}. Expected 1D array or 2D column vector.")


        # Get predictions from all base models
        # preds array shape: (n_models, n_samples)
        preds_list = []
        for name in self.model_names:
            # Base models should predict on NumPy array
            model_preds = self.models_dict[name].predict(X_np) 
            if model_preds.ndim == 1:
                preds_list.append(model_preds) # (n_samples,)
            elif model_preds.ndim == 2 and model_preds.shape[1] == 1:
                preds_list.append(model_preds.ravel()) # (n_samples,)
            else:
                raise ValueError(f"Model {name} predictions have incorrect shape: {model_preds.shape}")
        
        preds = np.array(preds_list) # (n_models, n_samples)

        # Initialize sigmas_sq (variances) for each model
        # Using mean squared error of each model as initial estimate
        current_sigmas_sq = np.array([np.mean(np.square(y_np - preds[k])) for k in range(self.n_models)])
        current_sigmas_sq = np.maximum(current_sigmas_sq, min_variance)

        current_weights = np.copy(self.weights_)
        
        last_iteration = 0 # To correctly print iteration count if max_iter is reached

        for iteration in range(max_iter):
            last_iteration = iteration
            prev_weights = np.copy(current_weights)

            # --- E-step: Compute responsibilities ---
            # errors_sq shape: (n_models, n_samples)
            errors_sq = np.square(y_col.T - preds) # y_col.T is (1, n_samples)
            
            # Unweighted Gaussian PDFs: N(y_i | f_ki, sigma_k^2)
            # pdf_components shape: (n_models, n_samples)
            pdf_components = self._gaussian_pdf(y_col.T, preds, current_sigmas_sq[:, np.newaxis])
            
            # Weighted PDFs: w_k * N(y_i | f_ki, sigma_k^2)
            # weighted_pdf_components shape: (n_models, n_samples)
            weighted_pdf_components = current_weights[:, np.newaxis] * pdf_components
            
            # Sum of weighted PDFs over models (denominator for responsibilities)
            # sum_weighted_pdfs shape: (n_samples,)
            sum_weighted_pdfs = np.sum(weighted_pdf_components, axis=0)
            
            # Responsibilities: r_ik = (w_k * N(y_i | f_ki, sigma_k^2)) / sum_j(w_j * N(y_i | f_ji, sigma_j^2))
            # responsibilities shape: (n_models, n_samples)
            responsibilities = weighted_pdf_components / (sum_weighted_pdfs[np.newaxis, :] + min_variance) # Add min_variance for stability
            
            # --- M-step: Update weights and variances ---
            # Sum of responsibilities for each model k over all samples
            # N_k shape: (n_models,)
            N_k = np.sum(responsibilities, axis=1)
            
            # Update weights
            # Add small epsilon to n_samples if it could be zero, though checked earlier
            current_weights = N_k / (n_samples + min_variance) 
            current_weights = current_weights / (np.sum(current_weights) + min_variance) # Ensure sum to 1

            # Update variances (sigma_k^2)
            current_sigmas_sq = np.sum(responsibilities * errors_sq, axis=1) / (N_k + min_variance) 
            current_sigmas_sq = np.maximum(current_sigmas_sq, min_variance)

            # Check for convergence
            if np.linalg.norm(current_weights - prev_weights) < tol:
                print(f"Converged at iteration {iteration + 1}.")
                break
        
        self.weights_ = current_weights
        self.sigmas_sq_ = current_sigmas_sq
        
        if last_iteration == max_iter - 1: # Check if loop finished due to max_iter
            if not (np.linalg.norm(current_weights - prev_weights) < tol) : # and not converged
                 print(f"EM algorithm reached max_iter ({max_iter}) without full convergence based on tol={tol}.")
        print(f"Final weights: {self.weights_}")
        print(f"Final model error variances (sigmas_sq): {self.sigmas_sq_}")

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict target values using the fitted BMA ensemble.
        This provides the expected value of the BMA predictive distribution.

        Args:
            X (Any): Input features (n_samples, n_features). Can be pandas DataFrame or NumPy array.


        Returns:
            np.ndarray: Ensemble predictions (n_samples,).
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_np = X.to_numpy()
        else:
            X_np = np.asarray(X)
        
        if X_np.ndim == 1: # Handle 1D array as a single sample with multiple features
            if self.models_dict: # Check if models_dict is populated
                # Get expected n_features from the first model if possible
                # This is a bit heuristic; ideally, the class would know n_features_in_
                # For now, assume if X_np is 1D, it's a single sample
                X_np = X_np.reshape(1, -1)


        if X_np.shape[0] == 0:
            return np.array([])
            
        preds_list = []
        for name in self.model_names:
            model_preds = self.models_dict[name].predict(X_np) # Predict on NumPy array
            if model_preds.ndim == 1:
                preds_list.append(model_preds)
            elif model_preds.ndim == 2 and model_preds.shape[1] == 1:
                preds_list.append(model_preds.ravel())
            else:
                raise ValueError(f"Model {name} predictions have incorrect shape ({model_preds.shape}) for X_np shape {X_np.shape}")
        
        all_preds = np.array(preds_list) # (n_models, n_samples)
        
        # Ensure all_preds is (n_models, n_samples)
        if X_np.shape[0] == 1 and all_preds.ndim == 1 and all_preds.shape[0] == self.n_models:
            # This happens if each model.predict(single_sample_X) returns a scalar,
            # and preds_list becomes a list of scalars. np.array(preds_list) becomes 1D.
            all_preds = all_preds.reshape(self.n_models, 1)
        elif all_preds.shape[1] != X_np.shape[0] and all_preds.shape[0] == X_np.shape[0]:
             # If models return (n_samples, 1) and list comprehension makes (n_models, n_samples, 1)
             # or if predict directly gives (n_samples, n_models)
             if all_preds.ndim == 2 and all_preds.shape[0] == self.n_models and all_preds.shape[1] == X_np.shape[0]:
                 pass # Correct shape (n_models, n_samples)
             elif all_preds.ndim == 2 and all_preds.shape[1] == self.n_models and all_preds.shape[0] == X_np.shape[0]:
                 all_preds = all_preds.T # Transpose (n_samples, n_models) to (n_models, n_samples)
             else:
                raise ValueError(f"Prediction array shape mismatch after potential transpose. all_preds: {all_preds.shape}, expected samples: {X_np.shape[0]}")


        ensemble_prediction = np.average(all_preds, axis=0, weights=self.weights_)
        return ensemble_prediction

    def predict_variance(self, X: Any) -> np.ndarray:
        """
        Predict the variance of the BMA ensemble prediction.
        The BMA variance is composed of between-model variance and within-model variance.

        Args:
            X (Any): Input features (n_samples, n_features). Can be pandas DataFrame or NumPy array.

        Returns:
            np.ndarray: Predicted variance for each sample (n_samples,).
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_np = X.to_numpy()
        else:
            X_np = np.asarray(X)
        
        if X_np.ndim == 1: # Handle 1D array as a single sample with multiple features
            if self.models_dict:
                 X_np = X_np.reshape(1, -1)

        if X_np.shape[0] == 0:
            return np.array([])

        preds_list = []
        for name in self.model_names:
            model_preds = self.models_dict[name].predict(X_np) # Predict on NumPy array
            if model_preds.ndim == 1:
                preds_list.append(model_preds)
            elif model_preds.ndim == 2 and model_preds.shape[1] == 1:
                preds_list.append(model_preds.ravel())
            else:
                raise ValueError(f"Model {name} predictions have incorrect shape ({model_preds.shape}) for X_np shape {X_np.shape}")

        all_preds = np.array(preds_list) # (n_models, n_samples)

        if X_np.shape[0] == 1 and all_preds.ndim == 1 and all_preds.shape[0] == self.n_models:
            all_preds = all_preds.reshape(self.n_models, 1)
        elif all_preds.shape[1] != X_np.shape[0] and all_preds.shape[0] == X_np.shape[0]:
             if all_preds.ndim == 2 and all_preds.shape[0] == self.n_models and all_preds.shape[1] == X_np.shape[0]:
                 pass
             elif all_preds.ndim == 2 and all_preds.shape[1] == self.n_models and all_preds.shape[0] == X_np.shape[0]:
                 all_preds = all_preds.T 
             else:
                raise ValueError(f"Prediction array shape mismatch after potential transpose in predict_variance. all_preds: {all_preds.shape}, expected samples: {X_np.shape[0]}")
        
        ensemble_mean = np.average(all_preds, axis=0, weights=self.weights_) # (n_samples,)
        
        squared_diff_from_mean = np.square(all_preds - ensemble_mean[np.newaxis, :])
        between_model_variance = np.sum(self.weights_[:, np.newaxis] * squared_diff_from_mean, axis=0) 
        
        within_model_variance = np.sum(self.weights_ * self.sigmas_sq_) 
        
        total_variance = between_model_variance + within_model_variance 
        return total_variance