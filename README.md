This is the code release for "Conformal Prediction for Long-Tailed Classification"


## Setting up virtual environment

Run
```
conda create --name ltc-env
conda activate ltc-env
conda install --yes --file requirements.txt
```

To make the environment accessible from Jupyter notebooks, run

```
ipython3 kernel install --user --name=ltc-env
```
This adds a kernel called `ltc-env` to your list of Jupyter kernels. 

## folders.json

Fill in this folder with the appropriate paths.

{
  "plantnet_data": [path to downloaded Pl@nt300-K data],
  "inaturalist_data": [path to downloaded Pl@nt300-K data. This should contain the files `train2018.json`, `val2018.json`, and sub-directory `train_val2018`],
  "scores_and_labels": [path to folder where model weights and softmax scores will be saved],
  "conformal_results":   [path to folder where conformal results from running `get_results.py` will be saved], 
  "figs": [path to folder where you want figures to be saved]
}

## Acknowledgements 

The code for implementing existing conformal procedures draws upon https://github.com/tiffanyding/class-conditional-conformal.
The code for processing datasets is based on https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py.