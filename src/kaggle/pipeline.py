from typing import Dict

from kaggle.pipelines import train_feature as trf
from kaggle.pipelines import test_feature as tef
from kaggle.pipelines import model
from kedro.pipeline import Pipeline

def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    trf_pipeline = trf.create_pipeline()
    tef_pipeline = tef.create_pipeline()
    model_pipeline = model.create_pipeline()
    return {
        "train_feature": trf_pipeline,
        "test_feature": tef_pipeline,
        "model": model_pipeline,
        "__default__": trf_pipeline + tef_pipeline + model_pipeline,
    }
