#!/bin/bash

nohup mlflow server -h 0.0.0.0 --default-artifact-root file://$(pwd)/mlruns &
nohup kedro jupyter notebook --ip="*" --no-browser --allow-root --NotebookApp.token='' &
nohup kedro viz --host 0.0.0.0 &
