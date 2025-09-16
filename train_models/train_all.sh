# # Train default models
# sbatch train.sbatch run_train.py plantnet
# sbatch train.sbatch run_train.py plantnet --trunc
# sbatch train.sbatch run_train.py inaturalist
# sbatch train.sbatch run_train.py inaturalist --trunc

# # Train models where 30% of val is used as proper validation and
# # the remaining 70% is untouched for conformal calibration 
# sbatch train.sbatch run_train.py plantnet --proper_cal
# sbatch train.sbatch run_train.py inaturalist --proper_cal
sbatch train.sbatch run_train.py plantnet --proper_cal --trunc
sbatch train.sbatch run_train.py inaturalist --proper_cal --trunc

# Train models using focal loss
# sbatch train.sbatch run_train.py plantnet --loss focal 
# sbatch train.sbatch run_train.py inaturalist--loss focal 
sbatch train.sbatch run_train.py plantnet --proper_cal --loss focal 
sbatch train.sbatch run_train.py inaturalist --proper_cal --loss focal 