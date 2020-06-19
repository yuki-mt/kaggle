import pandas as pd
import numpy as np

ENC_COL_FMT = '{}_target_enc'

def start_test(test_df: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
    test_df = pd.merge(test_df, item_df, on='item_id')
    test_df.drop('item_name', axis=1, inplace=True)
    return test_df

def encode_test(col: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    enc_col = ENC_COL_FMT.format(col)
    enc_map = train_df.groupby(col)[enc_col].mean().to_dict()

    feature = test_df[col].map(enc_map)
    feature.rename(enc_col, inplace=True)
    return feature

def pricing_map(train_df: pd.DataFrame, test_df: pd.DataFrame):
    category_map = train_df.groupby('item_category_id').item_price.mean().to_dict()
    item_map = train_df.groupby('item_id').item_price.mean().to_dict()
    shop_item_s = train_df.groupby(['shop_id', 'item_id']).item_price.mean()
    shop_item_s.index = shop_item_s.index.map(lambda x: f'{x[0]}-{x[1]}')
    shop_item_map = shop_item_s.to_dict()

    df = pd.DataFrame(index=test_df.index)

    test_shop_item_s = test_df.shop_id.astype(str) + '-' + test_df.item_id.astype(str)
    df['both_price'] = test_shop_item_s.map(shop_item_map)
    df.loc[df.both_price.isnull(), 'both_price'] = test_df.item_id.map(item_map)
    df.loc[df.both_price.isnull(), 'both_price'] = test_df.item_category_id.map(category_map)

    df['item_price'] = test_df.item_id.map(item_map)
    df.loc[df.item_price.isnull(), 'item_price'] = test_df.item_category_id.map(category_map)

    return df

def join_test(test_df: pd.DataFrame, price_df: pd.DataFrame, *enc_features):
    return _downcast_dtypes(pd.concat([test_df, price_df] + list(enc_features), axis=1))

def lag_test(lag_df: pd.DataFrame, df: pd.DataFrame):
    base_cols = lag_df.columns.difference(['shop_id', 'item_id'])
    shop_cols = list(base_cols[base_cols.str.contains('shop')])
    item_cols = list(base_cols[base_cols.str.contains('item')])
    both_cols = list(set(base_cols) - set(shop_cols) - set(item_cols))
    both_df = lag_df[both_cols + ['shop_id', 'item_id']]
    shop_df = lag_df[shop_cols + ['shop_id']].drop_duplicates(subset='shop_id')
    item_df = lag_df[item_cols + ['item_id']].drop_duplicates(subset='item_id')
    df = pd.merge(df, both_df, how='left', on=['shop_id', 'item_id'])
    df = pd.merge(df, shop_df, how='left', on='shop_id')
    df = pd.merge(df, item_df, how='left', on='item_id')
    return _downcast_dtypes(df.fillna(0))

def _downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype == "int64"]

    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)

    return df
