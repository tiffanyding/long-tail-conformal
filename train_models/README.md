To train the models, you must first download the datasets

## Downloading Pl@ntNet-300K

The dataset is available at https://zenodo.org/records/5645731#.Yuehg3ZBxPY

## Downloading iNaturalist

Download and untar the 2018 train-val data from https://github.com/visipedia/inat_comp/tree/master/2018#Data by running

Run the following to
1. Download and untar the 2018 train-val data
2. Download and untar the train and val annotations

```
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz
tar -xzvf train_val2018.tar.gz

wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.json.tar.gz
tar -xzvf train2018.json.tar.gz

wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/val2018.json.tar.gz
tar -xzvf val2018.json.tar.gz
```

## Set up virtual environment for training models

Run
```
conda create --name train-env python=3.12.8
conda activate train-env
conda install --yes --file train-requirements.txt
```

## Training the models and generating softmax scores

To train on full PlantNet-300k:

```
        python run_train.py plantnet
```

To train on truncated PlantNet-300k:

```
        python run_train.py plantnet --trunc
```

To train on full iNaturalist:

```
        python run_train.py inaturalist
```

To train on truncated PlantNet-300k:

```
        python run_train.py inaturalist --trunc
```

The `run_train.py` script will train a ResNet-50 for 20 epochs by default (this can be changed using the `--num-epochs` flag) and then save the weights from the epoch with the highest validation accuracy. Upon completing training, it will compute the softmax scores for the validation and test sets. This is saved to the `"scores_and_labels"` folder specified in `folders.json`. The train, validation, and test labels are also saved to this folder.