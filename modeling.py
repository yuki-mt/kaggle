from sklearn.base import BaseEstimator
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
from mlens.ensemble import SuperLearner
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.metrics import mean_squared_error

def get_model(param: dict) -> BaseEstimator:
    model_name = param.pop('name')
    if model_name == 'xgb':
        return XGBRegressor(**param[model_name])
    elif model_name == 'lgb':
        return LGBMRegressor(**param[model_name])
    elif model_name == 'cb':
        return CatBoostRegressor(**param[model_name])
    elif model_name == 'rf':
        return RandomForestRegressor(**param[model_name])
    elif model_name == 'svm':
        return make_pipeline(StandardScaler(), SVR(**param[model_name]))
    elif model_name == 'knn':
        return make_pipeline(StandardScaler(), KNeighborsRegressor(**param[model_name]))
    elif model_name == 'mlp':
        return make_pipeline(StandardScaler(), MLPRegressor(**param[model_name]))
    elif model_name == 'vote':
        return VotingRegressor(estimators=[
            ('svm', get_model(dict(param, name='svm'))),
            ('rf', get_model(dict(param, name='rf'))),
            ('lgb', get_model(dict(param, name='lgb'))),
            ('knn', get_model(dict(param, name='knn'))),
        ])
    elif model_name == 'stack':
        model = SuperLearner(scorer=mean_squared_error, random_state=132)
        model.add([
            get_model(dict(param, name='svm')),
            get_model(dict(param, name='rf')),
            get_model(dict(param, name='lgb')),
            get_model(dict(param, name='knn')),
        ])
        model.add_meta(GradientBoostingRegressor(random_state=22))
        return model
    elif model_name == 'sk_stack':
        return StackingRegressor(
            estimators=[
                ('svm', get_model(dict(param, name='svm'))),
                ('rf', get_model(dict(param, name='rf'))),
                ('lgb', get_model(dict(param, name='lgb'))),
                ('knn', get_model(dict(param, name='knn'))),
            ],
            final_estimator=GradientBoostingRegressor(random_state=42)
        )
