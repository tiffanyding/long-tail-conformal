import argparse
import time

from train import *

'''
To train on full PlantNet-300k:

        python run_train.py plantnet

To train on truncated PlantNet-300k:

        python run_train.py plantnet --trunc

To train on full iNaturalist:

        python run_train.py inaturalist

To train on truncated PlantNet-300k:

        python run_train.py inaturalist --trunc
'''


if __name__ == "__main__":
    st = time.time()
    
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('dataset', type=str, choices=['plantnet', 'inaturalist'],
                        help='Name of the dataset to train model on')
    
    parser.add_argument('--trunc', dest='trunc', action='store_true', # whether to truncate the dataset
                    help='Set the flag value to True.')
    parser.set_defaults(trunc=False)

    parser.add_argument('--frac_val', type=float, default=0.1,
                        help='Fraction of data to reserve for validation')
    parser.add_argument('--num_epochs', type=int, default=20,
                    help='Number of epochs to train for')

    args = parser.parse_args()

    if args.trunc:
        dset_name = args.dataset + '-trunc'
    else:
        dset_name = args.dataset

    config = {
        'batch_size' : 32, 
        'lr' : 0.0001,
        'num_epochs' : args.num_epochs,
        'device' : 'cuda',
        'frac_val' : args.frac_val, 
        'num_workers' : 4,
        'dataset_name' : args.dataset,
        'truncate': args.trunc,
        'feature_extract': False, # Whether to only tune the final layer
        'model_filename' : f'best-{dset_name}-model',
    }
    
    
    config = postprocess_config(config)
    
    # get_model(config) # train model only
    get_val_test_softmax_and_labels(config) # train model and apply to val and test datasets

    print(f'Time taken: {(time.time() - st) / 60:.2f} minutes')