## setup

```
$ unzip competitive-data-science-predict-future-sales.zip -d data/01_raw/

$ sudo apt-get install -y gcc g++
$ conda env create -f=conda.yaml
$ conda activate kaggle

# run pipeline and create all artifacts
# you need 64GiB RAM
$ kedro run

# start jupyter notebook, MLFlow server and kedro-viz
$ ./setup.sh
```
