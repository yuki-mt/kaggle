from metaflow import Run
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import mlflow
import hydra
from omegaconf import DictConfig
from logging import getLogger
from typing import Tuple
from time import time
import os

from modeling import get_model

logger = getLogger(__name__)

def train(run: Run, param: DictConfig) -> Tuple[BaseEstimator, np.float]:
    model = get_model(dict(param))
    logger.info(f'model name is: {param.name}')
    model.fit(run.data.X_train, run.data.y_train)
    y_pred = model.predict(run.data.X_train)
    score = np.sqrt(mean_squared_error(run.data.y_train, y_pred))
    return model, score

def validate(model: BaseEstimator, run: Run) -> np.float:
    y_pred = model.predict(run.data.X_val)
    score = np.sqrt(mean_squared_error(run.data.y_val, y_pred))
    return score

def test(run: Run, param: DictConfig) -> pd.DataFrame:
    model = get_model(dict(param))
    X_train = pd.concat([run.data.X_train, run.data.X_val], ignore_index=True)
    y_train = pd.concat([run.data.y_train, run.data.y_val], ignore_index=True)
    model.fit(X_train, y_train)

    X_test = run.data.X_test
    id_col = X_test['ID'].reset_index(drop=True).astype(np.int32)
    y_pred = model.predict(X_test.drop(['ID'], axis=1))
    pred_col = pd.Series(y_pred, name='item_cnt_month').astype(np.float32)
    return pd.concat([id_col, pred_col], axis=1)


@hydra.main(config_path='config.yaml')
def main(cfg: DictConfig):
    start = time()
    run = Run(f'SalesFlow/{cfg.data.run_id}')

    logger.info('start training')
    model, score_train = train(run, cfg.model)
    logger.info('start validation')
    score_val = validate(model, run)
    logger.info('start test')
    test_df = test(run, cfg.model)
    logger.info('finish test')

    mlflow.set_tracking_uri('http://localhost:5000')
    experiment_name = 'coursera'
    mlflow.set_experiment(experiment_name)
    tracking = mlflow.tracking.MlflowClient()
    experiment = tracking.get_experiment_by_name(experiment_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.log_param("metaflow_run_id", cfg.data.run_id)
        model_param = dict(cfg.model)
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

        mlflow.log_metric("time", time() - start)


if __name__ == '__main__':
    main()
