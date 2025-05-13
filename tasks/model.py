from prefect import task
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tasks.ensemble import BMAEnsembleRegressor
import pandas as pd
import joblib

@task
def train_base_model(X: pd.DataFrame, y: pd.DataFrame)-> dict:
    models = {
        # "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        "CatBoost": CatBoostRegressor(n_estimators=100, random_state=42, verbose=0),
    }
    
    model_paths = {}
    for model_name, model in models.items():
        print(f"Training {model_name}... X shape: {X.shape}, y shape: {y.shape}")
        model.fit(X, y)
        path = f"output/models/{model_name}.joblib"
        joblib.dump(model, path)
        model_paths[model_name] = path
    return model_paths

@task
def train_em_ensemble(model_paths: dict[str, str], X_train, y_train):
    loaded_models = {name: joblib.load(path) for name, path in model_paths.items()}
    em_ensemble = BMAEnsembleRegressor(loaded_models)
    em_ensemble.fit(X_train, y_train)
    joblib.dump(em_ensemble, "output/models/em_ensemble.joblib")
    return "output/models/em_ensemble.joblib"


@task
def predict_em_ensemble(model_path: str, X_test):
    em_model = joblib.load(model_path)
    return em_model.predict(X_test)
