import pandas as pd
import numpy as np
from typing import List

ENC_COL_FMT = '{}_target_enc'

def start(df: pd.DataFrame) -> pd.DataFrame:
    df.drop_duplicates(inplace=True)
    df.loc[df.item_price < 0, 'item_price'] = df.item_price.median()
    df['sales_price'] = df['item_cnt_day'] * df['item_price']
    return df

def remove_outliner(df: pd.DataFrame) -> pd.DataFrame:
    def remove(df: pd.DataFrame, col: str):
        q = df[col].quantile(0.99)
        return df[df[col] < q]

    df = remove(df, 'item_price')
    df = remove(df, 'item_cnt_day')
    return df

def grid(grid_cols: List[str], df: pd.DataFrame) -> pd.DataFrame:
    from itertools import product

    grid = []
    for block_num in df['date_block_num'].unique():
        cur_shops = df[df['date_block_num']==block_num]['shop_id'].unique()
        cur_items = df[df['date_block_num']==block_num]['item_id'].unique()
        grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype='int32'))
    grid_df = pd.DataFrame(np.vstack(grid), columns=grid_cols, dtype=np.int32)

    return grid_df


def group(name: str, cols: List[str], df: pd.DataFrame) -> pd.DataFrame:
    gb = df.groupby(cols, as_index=False)
    target_name = 'target' if name == 'both' else f'target_{name}'
    df = gb.item_cnt_day.agg({target_name: 'sum'})
    if name != 'shop':
        df[f'{name}_price'] = gb.item_price.agg({'item_price': 'mean'}).item_price
    df[f'{name}_sales_price'] = gb.sales_price.agg({'sales_price': 'sum'}).sales_price
    return cols, df

def merge_grouped(grid_df: pd.DataFrame, *args) -> pd.DataFrame:
    merged_df = grid_df
    for cols, df in args:
        merged_df = pd.merge(merged_df, df, how='left', on=cols)

    merged_df = _downcast_dtypes(merged_df)
    merged_df.loc[merged_df.both_price.isnull(), 'both_price'] = merged_df.item_price
    merged_df.fillna(0, inplace=True)
    return merged_df

def calc_global_mean(df: pd.DataFrame) -> float:
    return df.target.mean()

def merge_category(df: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(df, item_df, on='item_id')
    df.drop('item_name', axis=1, inplace=True)
    return df

def mean_encoding(target_col: str, df: pd.DataFrame, global_mean: float) -> pd.Series:
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=132)
    X = df.drop(['target'], axis=1)
    y = df['target']
    feats = []
    for train_index, test_index in kf.split(X, y):
        train_df = df.iloc[train_index, :]
        test_df = df.iloc[test_index, :]
        feats.append(test_df[target_col].map(train_df.groupby(target_col).target.mean()))
    feature = pd.concat(feats).sort_index()
    feature.fillna(global_mean, inplace=True)
    feature.rename(ENC_COL_FMT.format(target_col), inplace=True)
    return feature

def join_encoding(df: pd.DataFrame, *features) -> pd.DataFrame:
    return _downcast_dtypes(pd.concat([df] + list(features), axis=1))

def cols_for_lag(category_cols: List[str], grid_cols: List[str], df: pd.DataFrame):
    lag_drop_cols = ['item_category_id'] \
        + [ENC_COL_FMT.format(c) for c in category_cols]\
        + ['both_price', 'item_price']

    cols_to_rename = list(df.columns.difference(grid_cols + lag_drop_cols))
    return lag_drop_cols, cols_to_rename

def lag(shift: int, train_shift: pd.DataFrame, cols_to_rename: List[str], lag_drop_cols: List[str]):
    train_shift['date_block_num'] = train_shift['date_block_num'] + shift
    train_shift.drop(lag_drop_cols, axis=1, inplace=True)

    def rename_lag(x: str) -> str:
        return '{}_lag_{}'.format(x, shift) if x in cols_to_rename else x

    return train_shift.rename(columns=rename_lag)

def join_lag(min_block_num: int, grid_cols: List[str], df: pd.DataFrame, npartitions: int, *shifts):
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar

    df = df[df['date_block_num'] >= min_block_num]
    test_block = df['date_block_num'].max() + 1
    test_ddf = dd.from_pandas(pd.DataFrame(), npartitions=npartitions)
    ddf = dd.from_pandas(df, npartitions=npartitions)
    for s in shifts:
        base_cols = s.columns.difference(grid_cols)
        shop_cols = list(base_cols[base_cols.str.contains('shop')])
        shop_keys = list(set(grid_cols) - set(['item_id']))
        item_cols = list(base_cols[base_cols.str.contains('item')])
        item_keys = list(set(grid_cols) - set(['shop_id']))
        both_cols = list(set(base_cols) - set(shop_cols) - set(item_cols))
        both_ddf = dd.from_pandas(s[both_cols + grid_cols], npartitions=npartitions)
        shop_ddf = dd.from_pandas(s[shop_cols + shop_keys], npartitions=npartitions)\
            .drop_duplicates(subset=shop_keys)
        item_ddf = dd.from_pandas(s[item_cols + item_keys], npartitions=npartitions)\
            .drop_duplicates(subset=item_keys)
        ddf = dd.merge(ddf, both_ddf, how='left', on=grid_cols)
        ddf = dd.merge(ddf, shop_ddf, how='left', on=shop_keys)
        ddf = dd.merge(ddf, item_ddf, how='left', on=item_keys)

        _test_df = s[s.date_block_num == test_block].drop(['date_block_num'], axis=1)
        _test_ddf = dd.from_pandas(_test_df, npartitions=npartitions)
        if len(test_ddf) == 0:
            test_ddf = _test_ddf
        else:
            test_ddf = dd.merge(test_ddf, _test_ddf, how='outer',
                                on=list(set(grid_cols) - set(['date_block_num'])))
    with ProgressBar():
        df = ddf.compute(scheduler='processes').fillna(0)
        test_df = test_ddf.compute(scheduler='processes').fillna(0)
    return df, test_df

def remove_cols(shift_range: List[str], grid_cols: List[str], df: pd.DataFrame, lag_drop_cols: List[str]):
    lag_cols = [col for col in df.columns if col[-1] in [str(item) for item in shift_range]]

    keep_cols = list(set(grid_cols) - set(['date_block_num'])) + lag_drop_cols + lag_cols

    return df.date_block_num, df[keep_cols], df.target

def split_data(date_blocks: pd.Series, X: pd.DataFrame, y: pd.Series):
    val_block = date_blocks.max()
    return (
        X.loc[date_blocks < val_block],
        y.loc[date_blocks < val_block],
        X.loc[date_blocks == val_block],
        y.loc[date_blocks == val_block]
    )

def _downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype == "int64"]

    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)

    return df
