"""Small classical ML comparison on the simulated user-rate target."""

from pathlib import Path
from time import perf_counter
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.data_pipeline import FEATURE_COLUMNS


def _model_pipeline(model) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])


def compare_regressors(train_frame: pd.DataFrame, test_frame: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    x_train, y_train = train_frame[FEATURE_COLUMNS], train_frame["rate_target_bps"] / 1e6
    x_test, y_test = test_frame[FEATURE_COLUMNS], test_frame["rate_target_bps"] / 1e6
    models = {
        "Linear Regression": _model_pipeline(LinearRegression()),
        "Ridge Regression": _model_pipeline(Ridge(alpha=1.0)),
        "Lasso Regression": _model_pipeline(Lasso(alpha=0.01, max_iter=5000, random_state=42)),
        "Decision Tree": _model_pipeline(DecisionTreeRegressor(max_depth=6, random_state=42)),
        "Random Forest": _model_pipeline(RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=1)),
    }
    rows = []
    for name, model in models.items():
        start = perf_counter()
        model.fit(x_train, y_train)
        train_ms = (perf_counter() - start) * 1000.0
        start = perf_counter()
        predictions = model.predict(x_test)
        inference_ms = (perf_counter() - start) * 1000.0
        mse = mean_squared_error(y_test, predictions)
        rows.append({
            "model": name, "task": "user rate regression", "split": "scenario test",
            "mae_mbps": mean_absolute_error(y_test, predictions), "mse_mbps2": mse,
            "rmse_mbps": np.sqrt(mse), "r2": r2_score(y_test, predictions),
            "training_time_ms": train_ms, "inference_time_ms": inference_ms,
        })
        joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS, "target_unit": "Mbps"}, output / (name.lower().replace(" ", "_") + ".joblib"))
        if name in {"Decision Tree", "Random Forest"}:
            estimator = model.named_steps["model"]
            pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": estimator.feature_importances_}).to_csv(
                output / (name.lower().replace(" ", "_") + "_feature_importance.csv"), index=False
            )
    result = pd.DataFrame(rows)
    result.to_csv(output / "classical_model_comparison.csv", index=False)
    return result


def predict_rate(model_artifact: str, frame: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(model_artifact)
    missing = sorted(set(bundle["feature_columns"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Inference data is missing features: {', '.join(missing)}")
    return np.asarray(bundle["model"].predict(frame[bundle["feature_columns"]]), dtype=float)
