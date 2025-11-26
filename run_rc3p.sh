# python get_results.py inaturalist-trunc --methods rc3p --override
# python get_results.py plantnet-trunc  --methods rc3p --override
# python get_results.py plantnet  --methods rc3p --override
python get_results.py inaturalist  --methods rc3p --override

python get_results.py plantnet  --methods rc3p --override --loss focal_loss
python get_results.py inaturalist  --methods rc3p --override --loss focal_loss