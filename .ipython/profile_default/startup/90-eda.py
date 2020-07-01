import pandas as pd
import sys
import numpy as np
from matplotlib import pyplot as plt
from pandas_profiling import ProfileReport
import umap
from scipy.sparse.csgraph import connected_components
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from typing import Optional, List, Dict, Union
import xgboost as xgb
import japanize_matplotlib
from collections import defaultdict
import statsmodels.api as sm
from scipy.stats import entropy

pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_rows", 200)


def plot_2d(X: pd.DataFrame, y: Optional[pd.Series] = None,
            method: str = 'umap', param: dict = {}):
    if method == 'umap':
        embedding = umap.UMAP(**param).fit_transform(X)
    elif method == 'tsne':
        embedding = TSNE(n_components=2, **param).fit_transform(X)
    else:
        raise ValueError('method can be either "umap" or "tsne"')

    if y is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], cmap=cm.bwr, alpha=0.5, marker='*', s=20)
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap=cm.bwr, alpha=0.5, marker='*', s=20)
        plt.colorbar()
    plt.show()

def select_features(X: pd.DataFrame, y: pd.Series, top_n: int = 50,
                    multi_classes: bool = False,
                    is_regression: bool = False) -> List[str]:
    if is_regression:
        xgb_param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
        }
    elif multi_classes:
        xgb_param = {
            'objective': 'multi:softmax',
            'eval_metric': 'mlogloss',
        }
    else:
        xgb_param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
        }

    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(xgb_param, dtrain)
    xgb.plot_importance(model, max_num_features=top_n, importance_type='gain')

    feature_list = model.get_score(importance_type='gain')
    feature_list = sorted(feature_list.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [f for f, score in feature_list]
def hist_with_label(X: pd.DataFrame, y: pd.Series,
                    is_regression: bool = False, bins: Optional[int] = None):
    if is_regression:
        cut_y = pd.cut(y, 5).astype(str)
        labels = sorted(list(cut_y.unique()))
        y = cut_y
    else:
        labels = sorted(list(y.unique()))
    for fname in X.columns:
        f_by_label = [X.loc[y == label, fname].values for label in labels]
        plt.hist(f_by_label, stacked=True, label=labels, bins=bins)
        plt.title(fname)
        plt.legend(loc='best')
        plt.show()

def hist_train_test(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series, is_regression: bool = False,
                    bins: Optional[int] = None):
    if is_regression:
        df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        for fname in df.columns:
            f_train = df[fname].values
            f_test = test_df[fname].values
            plt.figure()
            plt.hist(f_train, bins=bins, alpha=0.3, histtype='stepfilled', color='r', label='train')
            plt.hist(f_test, bins=bins, alpha=0.3, histtype='stepfilled', color='b', label='test')
            plt.legend(loc='best')
            plt.title(f'feature: {fname}')
            plt.show()
    else:
        for fname in X_train.columns:
            for label in y_train.unique():
                f_train = X_train.loc[y_train == label, fname].values
                f_test = X_test.loc[y_test == label, fname].values
                plt.figure()
                plt.hist(f_train, bins=bins, alpha=0.3, histtype='stepfilled', color='r', label='train')
                plt.hist(f_test, bins=bins, alpha=0.3, histtype='stepfilled', color='b', label='test')
                plt.legend(loc='best')
                plt.title(f'feature: {fname}, label: {label}')
                plt.show()

def remove_outlier(df: pd.DataFrame, col: str, quantile: float = 0.99):
    q = df[col].quantile(quantile)
    return df[df[col] < q]

def diff_train_test(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series, is_regression: bool = False):
    print('========shape=========')
    print('train shape:', X_train.shape)
    print('test shape:', X_test.shape)
    if not is_regression:
        print('========label count=========')
        print('- train')
        for label in y_train.unique():
            print(f'label: {label}. count: {len(y_train[y_train == label])}')
        print('- test')
        for label in y_test.unique():
            print(f'label: {label}. count: {len(y_test[y_test == label])}')
    print('========column difference=========')
    print('train - test: ', list(X_train.columns.difference(X_test.columns)))
    print('test - train: ', list(X_test.columns.difference(X_train.columns)))

def time_series(df: pd.DataFrame, features: List[str], time_col: str):
    values: Dict[str, List[float]] = defaultdict(list)
    for year in years:
        for month in range(1, 13):
            if year == years[-1] and month == last_month:
                break
            df = pd.read_csv(path_template.format(year, f'{year}{str(month).zfill(2)}'))
            for f in features:
                value = df[f].mean()
                if np.isnan(value):
                    print(f'value is nan. date: {year}-{month}, feature: {f}')
                    if len(values[f]) > 1:
                        values[f].append(values[f][-1])
                    else:
                        values[f].append(0)
                else:
                    values[f].append(value)
    for f in features:
        res = sm.tsa.seasonal_decompose(values[f], period=12, model="multiplicative")
        res.plot()

def hist_kl_divergence(s1: pd.Series, s2: pd.Series, bins: int = 20) -> float:
    contated = pd.concat([s1, s2], ignore_index=True)
    rng = (contated.min(), contated.max())
    dist1, _ = np.histogram(s1, bins=bins, range=rng)
    dist2, _ = np.histogram(s2, bins=bins, range=rng)
    epsilon = np.finfo(np.float32).eps
    dist1 = dist1.astype(np.float32) + epsilon
    dist2 = dist2.astype(np.float32) + epsilon
    return entropy(dist1, dist2)

def plot_kl_divergence(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, bins: int = 20):
    features = list(X_train.columns)
    train_val_kld = {}
    train_test_kld = {}
    for f in features:
        train_test_kld[f] = hist_kl_divergence(X_train[f], X_test[f], bins=bins)
        train_val_kld[f] = hist_kl_divergence(X_train[f], X_val[f], bins=bins)
    left = np.arange(len(features))
    height = 0.4
    plt.barh(left, [train_test_kld[f] for f in features], color='r', height=height, align='center', label='train_test')
    plt.barh(left + height, [train_val_kld[f] for f in features], color='b', height=height, align='center', label='train_val')
    plt.yticks(left + height / 2, features)
    plt.legend(loc='best')
    plt.show()

def basic_info(name_or_df: Union[str, pd.DataFrame]):
    if isinstance(name_or_df, str):
        df = catalog.load(name_or_df)
    else:
        df = name_or_df
    print('shape:', df.shape)
    print('--------------')
    print('unique numbers')
    print(df.nunique())
    print('--------------')
    print('num of duplicated:', len(df[df.duplicated()]))
    print('--------------')
    df.info()
    return df
