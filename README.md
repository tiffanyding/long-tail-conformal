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

## Getting started

TODO: provide code/example notebook for running Standard with PAS, Fuzzy CP, and Interp-Q.

## folders.json

Fill in this folder with the appropriate paths.

```
{
  "plantnet_data": [path to downloaded Pl@nt300-K data],
  "inaturalist_data": [path to downloaded Pl@nt300-K data. This should contain the files
                    `train2018.json`, `val2018.json`, and sub-directory `train_val2018`],
  "scores_and_labels": [path to folder where model weights and softmax scores will be saved],
  "conformal_results":   [path to folder where conformal results from running
                        `get_results.py` will be saved], 
  "figs": [path to folder where you want figures to be saved]
}
```

## Reproducing paper plots

To generate the plots from the paper:

**Step 0:** For each of the four datasets (`plantnet`, `plantnet-trunc`, `inaturalist`, and `inaturalist-trunc`), obtain the `val` and `test` softmax scores and labels, as well as the `train` labels and put them in the folder specified by `"scores_and_labels"` in `folders.json`. There are two options for doing this
- Option A: Follow the instructions in `train_models/README.md` to train the classifiers yourself
- Option B: Download the softmax scores we have precomputed, which are available at [TODO]

**Step 1:** Run `scripts/run_get_results.sh` by running  

```
sh scripts/run_get_results.sh
```
or, if you are using a Slurm system, by running
```
sbatch scripts/run_get_results.sh
```

This will apply various conformal prediction procedures to each dataset and save evaluation metrics to the `"conformal_results"` folder specified in `folders.json`.

**Step 2:** After saving these metrics, run the Jupyter notebooks in `notebooks/`. This will generate the figures and save them to the `"figs"` folder specified in `folders.json`.

## Acknowledgements 

The code for implementing existing conformal procedures draws upon https://github.com/tiffanyding/class-conditional-conformal.
The code for processing datasets is based on https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py.