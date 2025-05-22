#!/bin/bash
#SBATCH --job-name=get_results
#SBATCH -o LOG_get_results_%j.out
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00


# Main paper results
python get_results.py inaturalist-trunc
python get_results.py plantnet-trunc
python get_results.py plantnet
python get_results.py inaturalist
# Fuzzy CP with other projections
python get_results.py inaturalist-trunc --methods fuzzy-random fuzzy-quantile fuzzy-RErandom fuzzy-REquantile
python get_results.py plantnet-trunc --methods fuzzy-random fuzzy-quantile fuzzy-RErandom fuzzy-REquantile
python get_results.py plantnet --methods fuzzy-random fuzzy-quantile fuzzy-RErandom fuzzy-REquantile
python get_results.py inaturalist --methods fuzzy-random fuzzy-quantile fuzzy-RErandom fuzzy-REquantile
# Additional results
python get_results.py inaturalist-trunc --score PAS
python get_results.py plantnet-trunc --score PAS
python get_results.py plantnet --score PAS
python get_results.py inaturalist --score PAS

