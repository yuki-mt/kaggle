#!/bin/bash
nohup mlflow server -h 0.0.0.0 --default-artifact-root file://$(pwd)/mlruns &
nohup jupyter notebook --ip="*" --no-browser --allow-root --NotebookApp.token='' &
