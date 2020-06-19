from kedro.pipeline import Pipeline

from .nodes import train, validate, test, log
from kaggle import _node

def create_pipeline(**kwargs):
    nodes = [
        _node(train,
              ['x_train', 'y_train', 'params:model'],
              ['model', 'train_score']),
        _node(validate, ['x_val', 'y_val', 'model'], 'val_score'),
        _node(test,
              ['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'params:model'],
              'submission'),
        _node(log,
              ['submission', 'train_score', 'val_score', 'params:model'],
              None),
    ]
    return Pipeline(nodes)
