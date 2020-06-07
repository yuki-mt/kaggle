import optuna
import numpy as np
from metaflow import Run
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

run_id = '1591105226171747'
run = Run(f'SalesFlow/{run_id}')

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype == "int64"]

    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)

    return df


X_train = downcast_dtypes(run.data.X_train)
y_train = run.data.y_train
X_val = downcast_dtypes(run.data.X_val)
y_val = run.data.y_val

def get_score(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))

def objective_svm(trial):
    params = {
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'sigmoid']),
        'C': trial.suggest_loguniform('C', 1e-2, 1e+2),
        'gamma': trial.suggest_loguniform('gamma', 1e-2, 1e+1),
    }
    model = make_pipeline(StandardScaler(), SVR(**params))
    return get_score(model)

def objective_knn(trial):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
        'leaf_size': trial.suggest_int('leaf_size', 10, 100),
        'p': trial.suggest_int('p', 1, 2),
    }
    model = make_pipeline(StandardScaler(), KNeighborsRegressor(**params))
    return get_score(model)

def objective_lgb(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', -1, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 20, 400),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1e-3),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1e-3),
        'random_state': 132,
    }
    model = LGBMRegressor(**params)
    return get_score(model)

def objective_rf(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', -1, 500),
        'n_estimators': trial.suggest_int('n_estimators', 20, 400),
        'random_state': 132,
        'n_jobs': 16,
    }
    model = RandomForestRegressor(**params)
    return get_score(model)

def objective_cb(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', -1, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 20, 400),
        'random_state': 132,
    }
    model = CatBoostRegressor(**params)
    return get_score(model)

def objective_mlp(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', -1, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 20, 400),
        'random_state': 132,
    }
    model = make_pipeline(StandardScaler(), MLPRegressor(**params))
    return get_score(model)


if __name__ == '__main__':
    n_jobs = 30
    n_trials = 300
    storage = optuna.storages.RDBStorage(
        url='mysql://user:pw@localhost/db_name',
        engine_kwargs={
            'pool_size': n_jobs + 5,
            'max_overflow': 0
        }
    )
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(),
                                storage=storage,
                                study_name='coursera_lgb',
                                load_if_exists=True)
    study.optimize(objective_rf, n_trials=n_trials, n_jobs=n_jobs)
    print('params:', study.best_params)
