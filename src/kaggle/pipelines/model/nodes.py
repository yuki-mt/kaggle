from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import mlflow
from logging import getLogger
from typing import Tuple
from copy import deepcopy
import os

from .modeling import get_model

logger = getLogger(__name__)

def train(X: pd.DataFrame, y: pd.Series, param: dict) -> Tuple[BaseEstimator, np.float]:
    model = get_model(deepcopy(param))
    logger.info(f'model name is: {param["name"]}')
    model.fit(X, y)
    y_pred = model.predict(X)
    score = np.sqrt(mean_squared_error(y, y_pred))
    return model, score

def validate(X: pd.DataFrame, y: pd.Series, model: BaseEstimator) -> np.float:
    y_pred = model.predict(X)
    score = np.sqrt(mean_squared_error(y, y_pred))
    return score

def test(X_train: pd.DataFrame, y_train: pd.Series,
         X_val: pd.DataFrame, y_val: pd.Series,
         X_test: pd.DataFrame,
         param: dict) -> pd.DataFrame:
    model = get_model(param)
    X_train = pd.concat([X_train, X_val], ignore_index=True)
    y_train = pd.concat([y_train, y_val], ignore_index=True)
    model.fit(X_train, y_train)

    id_col = X_test['ID'].reset_index(drop=True).astype(np.int32)
    y_pred = model.predict(X_test.drop(['ID'], axis=1))
    pred_col = pd.Series(y_pred, name='item_cnt_month').astype(np.float32)
    return pd.concat([id_col, pred_col], axis=1)

def log(test_df: pd.DataFrame, score_train: np.float,
        score_val: np.float, model_param: dict):
    mlflow.set_tracking_uri('http://localhost:5000')
    experiment_name = 'coursera'
    mlflow.set_experiment(experiment_name)
    tracking = mlflow.tracking.MlflowClient()
    experiment = tracking.get_experiment_by_name(experiment_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        model_name = model_param.pop('name')
        for name, value in model_param[model_name].items():
            mlflow.log_param(f"model_{name}", value)
        mlflow.log_param("model_name", model_name)

        mlflow.log_metric("train_rsme", score_train)
        mlflow.log_metric("val_rsme", score_val)

        filename = 'submission.csv'
        test_df.to_csv(filename, index=False)
        mlflow.log_artifact(filename)
        os.remove(filename)
