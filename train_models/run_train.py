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

To train a model using focal loss, simply add `--loss focal'. For example, to train on PlantNet using the focal loss, run

        python run_train.py plantnet --loss focal

To split the validation set into a proper validation and proper calibration set, add `--proper_cal'. 
For example, to do this for Plantnet, run

        python run_train.py plantnet --proper_cal
'''

if __name__ == "__main__":
    st = time.time()
    
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('dataset', type=str, choices=['plantnet', 'inaturalist'],
                        help='Name of the dataset to train model on')
    
    parser.add_argument('--trunc', dest='trunc', action='store_true', # whether to truncate the dataset
                    help='Use this flag to truncate dataset')
    parser.set_defaults(trunc=False)
    
    parser.add_argument('--use_last_epoch', dest='use_last_epoch', action='store_true', # whether to truncate the dataset
                    help='Use this flag to generate softmax scores using the last epoch model (rather than select based on validation accuracy)')
    parser.set_defaults(use_last_epoch=False)

    parser.add_argument('--proper_cal', dest='proper_cal', action='store_true', # whether to truncate the dataset
                    help='Use this flag to do a 4-way data split where 30% of the conformal calibration dataset is set aside' +
                       'to use as a proper validation. The remaining 70% is untouched. Without this flag, the data for' +
                       'model validation and conformal calibration is the same')
    parser.set_defaults(proper_cal=False)
    
    parser.add_argument('--num_epochs', type=int, default=20,
                    help='Number of epochs to train for')
    parser.add_argument('--loss', type=str, default='cross_entropy',
                    help='Loss function: Options are "cross_entropy" or "focal" (designed for imbalanced data)')
    

    args = parser.parse_args()

    if args.trunc:
        dset_name = args.dataset + '-trunc'
    else:
        dset_name = args.dataset

    assert not (args.use_last_epoch and args.proper_cal), 'If using a proper calibration set,' + \
        'you should use the validation set for selecting the epoch'

    if args.use_last_epoch:
        filename = f'last-epoch-{dset_name}-model'
    elif args.proper_cal:
        filename = f'proper-cal-{dset_name}-model'
    else:
        filename = f'best-{dset_name}-model'

    config = {
        'batch_size' : 32, 
        'lr' : 0.0001,
        'num_epochs' : args.num_epochs,
        'device' : 'cuda',
        'num_workers' : 4,
        'dataset_name' : args.dataset,
        'truncate': args.trunc,
        'loss': args.loss,
        'feature_extract': False, # Whether to only tune the final layer
        'use_last_epoch': args.use_last_epoch,
        'proper_cal': args.proper_cal,
        'model_filename' : filename
    }
    
    
    config = postprocess_config(config)
    
    # get_model(config) # train model only
    get_val_test_softmax_and_labels(config) # train model and apply to val/cal and test datasets

    print(f'Time taken: {(time.time() - st) / 60:.2f} minutes')