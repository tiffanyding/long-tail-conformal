#!/bin/bash
#SBATCH --job-name=get_results
#SBATCH -o LOG_get_results_%j.out
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00


## Main results
# python get_results.py inaturalist-trunc
# python get_results.py plantnet-trunc
# python get_results.py plantnet
# python get_results.py inaturalist

# ## Additional results

# # Fuzzy CP with other projections
# python get_results.py inaturalist-trunc --methods fuzzy-random fuzzy-quantile fuzzy-RErandom fuzzy-REquantile
# python get_results.py plantnet-trunc --methods fuzzy-random fuzzy-quantile fuzzy-RErandom fuzzy-REquantile
# python get_results.py plantnet --methods fuzzy-random fuzzy-quantile fuzzy-RErandom fuzzy-REquantile
# python get_results.py inaturalist --methods fuzzy-random fuzzy-quantile fuzzy-RErandom fuzzy-REquantile

# # PAS score combined with other methods
# python get_results.py inaturalist-trunc --score PAS
# python get_results.py plantnet-trunc --score PAS
# python get_results.py plantnet --score PAS
# python get_results.py inaturalist --score PAS

# # # Focal loss (as of 9/15/25, had to retrain)
# # python get_results.py plantnet --loss focal_loss
# # python get_results.py inaturalist --loss focal_loss

# # 4-way data split (30% of val set is used as proper validation and 70% is used for conformal calibration)
# python get_results.py plantnet --model_type proper_cal
# python get_results.py inaturalist --model_type proper_cal
python get_results.py plantnet-trunc --model_type proper_cal
python get_results.py inaturalist-trunc --model_type proper_cal
