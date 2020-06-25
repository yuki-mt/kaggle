from kedro.pipeline import Pipeline

from .nodes import start, remove_outlier, grid, group, \
    merge_grouped, merge_category, calc_global_mean, \
    mean_encoding, join_encoding, cols_for_lag, lag, join_lag, \
    remove_cols, split_data

from kaggle import _partial, _node

grid_cols = ['shop_id', 'item_id', 'date_block_num']
group_cols = [
    ('both', grid_cols),
    ('shop', ['date_block_num', 'shop_id']),
    ('item', ['date_block_num', 'item_id']),
]
category_cols = ['shop_id', 'item_id', 'item_category_id']
shift_range = [1, 2, 3, 4, 5, 12]

def create_pipeline(**kwargs):
    nodes = [
        _node(start, 'sales_df', 'start_df'),
        _node(remove_outlier, 'start_df', 'preprocessed_df'),
        _node(_partial(grid, grid_cols), 'preprocessed_df', 'grid_df'),
        _node(merge_grouped,
              ['grid_df'] + [f'{g[0]}_cols_and_df' for g in group_cols],
              'grid_merged_df'),
        _node(calc_global_mean, 'grid_merged_df', 'global_mean'),
        _node(merge_category, ['grid_merged_df', 'item_df'], 'cate_df'),
        _node(join_encoding,
              ['cate_df'] + [f'{col}_enc_feature' for col in category_cols],
              'train_enc_df'),
        _node(_partial(cols_for_lag, category_cols, grid_cols),
              'train_enc_df',
              ['lag_drop_cols', 'cols_to_rename']),
        _node(_partial(join_lag, max(shift_range), grid_cols),
              ['train_enc_df', 'params:npartitions'] + [f'shift{s}_df' for s in shift_range],
              ['lag_df', 'test_lag_df']),
        _node(_partial(remove_cols, shift_range, grid_cols),
              ['lag_df', 'lag_drop_cols'],
              ['date_blocks', 'train_x', 'train_y']),
        _node(split_data,
              ['date_blocks', 'train_x', 'train_y'],
              ['x_train', 'y_train', 'x_val', 'y_val']),
    ]
    for name, cols in group_cols:
        nodes.append(_node(
            _partial(group, name, cols),
            'preprocessed_df',
            f'{name}_cols_and_df',
            f'_{name}'
        ))
    for col in category_cols:
        nodes.append(_node(
            _partial(mean_encoding, col),
            ['cate_df', 'global_mean'],
            f'{col}_enc_feature',
            f'_{col}'
        ))
    for s in shift_range:
        nodes.append(_node(
            _partial(lag, s),
            ['train_enc_df', 'cols_to_rename', 'lag_drop_cols'],
            f'shift{s}_df',
            f'_{s}'
        ))
    return Pipeline(nodes)
