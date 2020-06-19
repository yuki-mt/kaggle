from kedro.pipeline import Pipeline

from .nodes import start_test, encode_test, \
    pricing_map, join_test, lag_test
from kaggle import _partial, _node

category_cols = ['shop_id', 'item_id', 'item_category_id']

def create_pipeline(**kwargs):
    nodes = [
        _node(start_test, ['test_df', 'item_df'], 'test_start_df'),
        _node(pricing_map, ['train_enc_df', 'test_start_df'], 'price_df'),
        _node(join_test,
              ['test_start_df', 'price_df'] + [f'{col}_test_enc_feature' for col in category_cols],
              'test_join_df'),
        _node(lag_test,
              ['test_lag_df', 'test_join_df'],
              'x_test'),
    ]
    for col in category_cols:
        nodes.append(_node(
            _partial(encode_test, col),
            ['train_enc_df', 'test_start_df'],
            f'{col}_test_enc_feature',
            f'_{col}'
        ))
    return Pipeline(nodes)
