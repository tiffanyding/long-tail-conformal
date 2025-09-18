import sys; sys.path.append("../") # For relative imports


import copy
import json
import numpy as np
import os
import pickle
import time
import torch
import traceback


from collections import Counter
from PIL import Image
from scipy.special import softmax
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Subset, Dataset, ConcatDataset
from tqdm import tqdm

import pdb

from utils.experiment_utils import get_plantnet_folder, get_inaturalist_folder, get_inputs_folder


# ------------------------------------------------
#                  Dataloaders
# ------------------------------------------------

# ------------------ PlantNet --------------------

class PlantNet(ImageFolder):
    def __init__(self, root, split, **kwargs):
        self.root = root
        self.split = split
        super().__init__(self.split_folder, **kwargs)

        # Copied from https://pytorch.org/vision/main/_modules/torchvision/datasets/inaturalist.html
        # to create index of all files: (full category id, filename)
        
        self.labels = []
        for dir_index, dir_name in enumerate(sorted(os.listdir(self.root))):
            files = os.listdir(os.path.join(self.root, dir_name))
            for _ in files:
                self.labels.append(dir_index)

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

# ------------------ iNaturalist 2018 --------------------

# Copied from https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py
def default_loader(path):
    return Image.open(path).convert('RGB')

# Copied from https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py
def load_taxonomy(ann_data, tax_levels, classes):
    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0]*len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic

# Copied, with modifications, from https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py
class iNaturalist(Dataset):
    def __init__(self, root, ann_file, is_train=True, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        # load annotations
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        self.ids = [aa['id'] for aa in ann_data['images']]

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            self.labels = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            self.labels = [0]*len(self.imgs)

        # load taxonomy
        self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                           #8142, 4412,    1120,     273,     57,      25,       6
        self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.labels)

        # print out some stats
        print ('\t' + str(len(self.imgs)) + ' images')
        print ('\t' + str(len(set(self.labels))) + ' classes')


        self.root = root
        self.is_train = is_train
        self.loader = default_loader


    def __getitem__(self, index):
        path = self.root + self.imgs[index]
        image = self.loader(path)
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_transformed_labels(self):
        if self.target_transform is None:
            return self.labels
        else:
            return [self.target_transform(l) for l in labels]

    def __len__(self):
        return len(self.ids)

# ------------------ General functions --------------------

def truncate_and_resplit_dataset(train_dataset, val_dataset, test_dataset=None, 
                                 num_test_samples=100, frac_val=0.1,
                                 return_label_arrays=False, print_info=True, seed=0):
    '''
    Remove all classes with fewer than num_test_samples. Then split data in the following way:
    - test will have num_test_samples per class
    - of the remaining samples per class, frac_val goes to val and the remaining goes to test
        (note this means that some classes may end up with 0 val examples)
    '''
    # Ensure that splits are the same each time we call this function
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Combine train, val, test into one dataset
    if test_dataset is not None:
        dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset, test_dataset])
    else:
        dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    
    # Get all labels as an array
    labels = []
    for ds in dataset.datasets:
        labels.extend(ds.labels)
    labels = np.array(labels)
    unique_classes = np.unique(labels)
    
    new_train_indices = []
    new_val_indices = []
    new_test_indices = []
    
    
    # Process each class separately
    kept_classes = []
    for cls in tqdm(unique_classes):
        # Find indices for the current class 
        cls_inds = np.where(labels == cls)[0]
    
        # Check if we have enough samples for this class (or if we should throw it out)
        if len(cls_inds) >= num_test_samples + 1:
            kept_classes.append(cls)
            
            np.random.shuffle(cls_inds)
            
            # Select up to 100 samples for the test dataset
            n_test = num_test_samples
            test_inds = cls_inds[:n_test]
            remaining_inds = cls_inds[n_test:]
            
            # From the remaining, assign 10% to validation (at least one sample) and 90% to training
            n_remaining = len(remaining_inds)
            # n_val = max(1, int(round(frac_val * n_remaining))) # requires at least one sample to go into n_val
            n_val = int(round(frac_val * n_remaining))
            n_train = n_remaining - n_val
        
            val_inds = remaining_inds[:n_val]
            train_inds = remaining_inds[n_val:]
            
            new_test_indices.extend(test_inds)
            new_val_indices.extend(val_inds)
            new_train_indices.extend(train_inds)
    
    # Create new datasets using the indices
    new_train_dataset = Subset(dataset, new_train_indices)
    new_val_dataset = Subset(dataset, new_val_indices)
    new_test_dataset = Subset(dataset, new_test_indices)
    
    # Map class labels to consecutive 0,1,2,...
    label_remapping = {}
    idx = 0
    for k in kept_classes:
        label_remapping[k] = idx
        idx += 1

    if print_info:
        print(f'Kept {len(kept_classes)} classes after filtering out classes with less than {num_test_samples+1} examples\n')
        
        print("New train dataset size:", len(new_train_dataset))
        print("New val dataset size:", len(new_val_dataset))
        print("New test dataset size:", len(new_test_dataset))

    if return_label_arrays:
        new_train_labels = labels[new_train_indices]
        new_val_labels = labels[new_val_indices]
        new_test_labels = labels[new_test_indices]

        # Apply label remapping
        new_train_labels = [label_remapping[k] for k in new_train_labels]
        new_val_labels = [label_remapping[k] for k in new_val_labels]
        new_test_labels = [label_remapping[k] for k in new_test_labels]
        
        return new_train_dataset, new_val_dataset, new_test_dataset, label_remapping, \
                new_train_labels, new_val_labels, new_test_labels

    return new_train_dataset, new_val_dataset, new_test_dataset, label_remapping


def get_datasets(dataset_name, truncate=False, root=None, return_labels=False, seed=0):

    # Ensure that splits are the same each time we call this function
    np.random.seed(seed)
    torch.manual_seed(seed)

    crop_size = 224
    image_size = 256
    
    if dataset_name == 'plantnet':
        if root is None:
            # root = '/home-warm/plantnet/plantnet_300K/images' # MODIFY FOR YOUR DEVICE
            root = get_plantnet_folder()

        # transforms copied from https://github.com/plantnet/PlantNet-300K/blob/main/utils.py#L180
        transform_train = transforms.Compose([transforms.Resize(size=image_size), transforms.RandomCrop(size=crop_size),
                                              transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                          std=[0.229, 0.224, 0.225])])
        transform_test = transforms.Compose([transforms.Resize(size=image_size), transforms.CenterCrop(size=crop_size),
                                             transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])])
        if truncate:
            
            pth = cache_folder + f'{dataset_name}-trunc_label_remapping.pkl'
            if os.path.isfile(pth):
                with open(pth, 'rb') as f:
                    label_remapping = pickle.load(f)
            else:
                      
                train_dataset = PlantNet(root, 'train')
                val_dataset = PlantNet(root, 'val')
                test_dataset = PlantNet(root, 'test')
                _, _, _, label_remapping = truncate_and_resplit_dataset(train_dataset, val_dataset, test_dataset, 
                                     num_test_samples=100, frac_val=0.1,
                                     return_label_arrays=False, print_info=False)
                with open(pth, 'wb') as f:
                    pickle.dump(label_remapping, f)
                    print('Saved label remapping after truncation to' + pth)
                     
            # Reload datasets with target_transform (to remap classes to consecutive 0,1,2,...)
            target_transform = lambda k: label_remapping[k]
                
            train_dataset = PlantNet(root, 'train', transform=transform_train, target_transform=target_transform)
            val_dataset = PlantNet(root, 'val', transform=transform_test, target_transform=target_transform)
            test_dataset = PlantNet(root, 'test', transform=transform_test, target_transform=target_transform)
            
            output = truncate_and_resplit_dataset(train_dataset, 
                                                     val_dataset, test_dataset, 
                                                     num_test_samples=100, frac_val=0.1,
                                                     return_label_arrays=True)
            train_dataset, val_dataset, test_dataset, _, train_labels, val_labels, test_labels = output
            
        else:   
            train_dataset = PlantNet(root, 'train', transform=transform_train)
            val_dataset = PlantNet(root, 'val', transform=transform_test)
            test_dataset = PlantNet(root, 'test', transform=transform_test)
            train_labels, val_labels, test_labels = train_dataset.labels, val_dataset.labels, test_dataset.labels

    elif dataset_name == 'inaturalist':
        if root is None:
            # root = '/home-warm/plantnet/inaturalist/'
            root = get_inaturalist_folder()
        dataset_root = root
        train_annot = f'{root}/train2018.json'
        val_annot = f'{root}/val2018.json'

        # standardization values copied from https://github.com/macaodha/inat_comp_2018/blob/master/inat2018_loader.py
        transform = transforms.Compose([transforms.Resize(size=image_size), transforms.CenterCrop(size=crop_size),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if truncate:

            pth = cache_folder + f'/{dataset_name}-trunc_label_remapping.pkl'
            os.makedirs(cache_folder, exist_ok=True)
            if os.path.isfile(pth):
                with open(pth, 'rb') as f:
                    label_remapping = pickle.load(f)
            else:
                      
                full_train_dataset = iNaturalist(dataset_root, train_annot, transform=transform)
                val_dataset = iNaturalist(dataset_root, val_annot, transform=transform)
                _, _, _, label_remapping = truncate_and_resplit_dataset(full_train_dataset, val_dataset, test_dataset=None, 
                                     num_test_samples=100, frac_val=0.1,
                                     return_label_arrays=False, print_info=False)
                with open(pth, 'wb') as f:
                    pickle.dump(label_remapping, f)
                    print('Saved label remapping after truncation to' + pth)
                    
            # Reload datasets with target_transform (to remap classes to consecutive 0,1,2,...)
            target_transform = lambda k: label_remapping[k]
            full_train_dataset = iNaturalist(dataset_root, train_annot, transform=transform, target_transform=target_transform)
            val_dataset = iNaturalist(dataset_root, val_annot, transform=transform, target_transform=target_transform)
            
            output = truncate_and_resplit_dataset(full_train_dataset, val_dataset, 
                                                 test_dataset=None, 
                                                 num_test_samples=100, frac_val=0.1,
                                                 return_label_arrays=True)
            train_dataset, val_dataset, test_dataset, _, train_labels, val_labels, test_labels = output
            
        else:   
            ## Create splits that are all representative samples
            # 1) load the two pools
            full_train_dataset = iNaturalist(dataset_root, train_annot, transform=transform)
            val_dataset_orig   = iNaturalist(dataset_root, val_annot,   transform=transform)
            
            # pull out their label arrays
            labels_full = np.array(full_train_dataset.labels)
            labels_val_orig = np.array(val_dataset_orig.labels)
            
            # 2) concat them
            combined_dataset = ConcatDataset([full_train_dataset, val_dataset_orig])
            labels_all = np.concatenate([labels_full, labels_val_orig])
            
            # 3) per‐class split indices
            frac_test = 0.1
            frac_val  = 0.1
            
            new_train_indices = []
            new_val_indices   = []
            new_test_indices  = []
            
            unique_classes = np.unique(labels_all)
            print("Splitting combined data into train/val/test (10% each per class)…")
            for cls in tqdm(unique_classes):
                cls_inds = np.where(labels_all == cls)[0].copy()
                np.random.shuffle(cls_inds)
            
                n = len(cls_inds)
                n_test = max(1, int(round(frac_test * n)))
                n_val  = max(1, int(round(frac_val  * n)))
            
                # ensure at least one left for train
                if n_test + n_val >= n:
                    n_test = min(n_test, n - 2)
                    n_val  = min(n_val,  n - 1 - n_test)
                    n_test = max(1, n_test)
                    n_val  = max(1, n_val)
            
                test_inds  = cls_inds[:n_test]
                val_inds   = cls_inds[n_test:n_test + n_val]
                train_inds = cls_inds[n_test + n_val:]
            
                new_test_indices .extend(test_inds.tolist())
                new_val_indices  .extend(val_inds.tolist())
                new_train_indices.extend(train_inds.tolist())
            
            # 4) wrap into Subsets
            train_dataset = Subset(combined_dataset, new_train_indices)
            val_dataset   = Subset(combined_dataset, new_val_indices)
            test_dataset  = Subset(combined_dataset, new_test_indices)
            
            # 5) Create a .labels attribute so we can still call .labels on each split
            train_dataset.labels = labels_all[new_train_indices]
            val_dataset.labels   = labels_all[new_val_indices]
            test_dataset.labels  = labels_all[new_test_indices]
            val_labels   = np.array(val_dataset.labels)
            train_labels = np.array(train_dataset.labels)
            test_labels  = np.array(test_dataset.labels)
                        
            # print(f" train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
                
    else:
        raise ValueError("Valid dataset_name values are 'plantnet' and 'inaturalist'")
    
    if return_labels:
        return train_dataset, val_dataset, test_dataset, train_labels, val_labels, test_labels
        
    return train_dataset, val_dataset, test_dataset

def get_dataloaders(config, root=None):

    train_dataset, val_dataset, test_dataset = get_datasets(config['dataset_name'], 
                                                            truncate=config['truncate'], root=root)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], 
                                           shuffle=True, num_workers=config['num_workers'])

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], 
                                          shuffle=True, num_workers=config['num_workers'])

    if config['proper_cal']: # Do 4-way datasplit by selecting a random 30% of val to become a proper validation set 
                             # The rest will be used as the conformal calibration set
                          # Note: Randomization is NOT within each class
        np.random.seed(0) # For reproducibility
        frac_val = 0.3 # (Fraction of calibration set to use as proper val)
        num_val_samples = int(np.floor(frac_val * len(val_dataset)))
        indices = np.arange(len(val_dataset))
        np.random.shuffle(indices)
        proper_val_indices = indices[:num_val_samples]
        proper_cal_indices = indices[num_val_samples:]
        
        # Create new datasets using the indices
        proper_val_dataset = Subset(val_dataset, proper_val_indices)
        proper_cal_dataset = Subset(val_dataset, proper_cal_indices)

        val_loader = torch.utils.data.DataLoader(proper_val_dataset, batch_size=config['batch_size'], 
                                                 shuffle=True, num_workers=config['num_workers'])
        cal_loader = torch.utils.data.DataLoader(proper_cal_dataset, batch_size=config['batch_size'], 
                                                 shuffle=True, num_workers=config['num_workers'])

        print(f"Train size: {len(train_dataset)} | "
          f"Val size: {len(proper_val_dataset)} | "
          f"Cal size: {len(proper_cal_dataset)} | "
          f"Test size: {len(test_dataset)}")
    
        return train_loader, val_loader, cal_loader, test_loader
        

    else:

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], 
                                                 shuffle=True, num_workers=config['num_workers'])

        print(f"Train size: {len(train_dataset)} | "
          f"Val size: {len(val_dataset)} | "
          f"Test size: {len(test_dataset)}")
    
        return train_loader, val_loader, test_loader
    
    # ------------------------------------------------
#                  Model training
# ------------------------------------------------

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

def train_model(model, train_loader, val_loader, config):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    save_every_epoch = True # save weights every epoch if accuracy is better than previous best
    
    set_parameter_requires_grad(model, config['feature_extract'])

    params_to_update = model.parameters()
    print("Params to learn:")
    if config['feature_extract']:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # The above prints show which layers are being optimized

    if config['loss'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif config['loss'] == 'focal':
        # From https://github.com/AdeelH/pytorch-multi-class-focal-loss
        criterion = torch.hub.load(
        	'adeelh/pytorch-multi-class-focal-loss',
        	model='FocalLoss',
        	alpha=None, # defaults to 1 
        	gamma=2,
        	reduction='mean',
        	force_reload=False
        )
        print('Training model with focal loss')
    else:
        raise ValueError("Invalid loss. Options are 'cross_entropy' and 'focal'")

    optimizer = optim.Adam(params_to_update, lr=config['lr'])

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(config['num_epochs']):
        print('Epoch {}/{}'.format(epoch, config['num_epochs'] - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
 
                inputs = inputs.to(config['device'])
                labels = labels.to(config['device'])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            # Save model weights every epoch (overwriting previous saves)
            if config['use_last_epoch']: 
                print(f'Saving epoch {epoch} model')
                torch.save(model.state_dict(), cache_folder + config['model_filename'] + '.pth')
    
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Save best model by validation accuracy
    if not config['use_last_epoch']: 
        # load best model weights
        model.load_state_dict(best_model_wts)
    
        # Save best model weights
        torch.save(best_model_wts, cache_folder + config['model_filename'] + '.pth')
        
    with open(cache_folder + config['model_filename'] + '-config.pkl', 'wb') as f:
        pickle.dump(config, f)

    return model, val_acc_history


def get_softmax_and_labels(dataloader, model, config):
    model.eval() 
    model.to(config['device']) # Use GPU when available to speed up computation
    
    softmax_arr = np.zeros((len(dataloader.dataset), config['num_classes']))
    labels_arr = np.zeros((len(dataloader.dataset),), dtype=int)
    j = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(config['device']) # Use GPU when available to speed up computation
            labels_arr[j:j+inputs.shape[0]] = labels.numpy()
    
            # Get model outputs
            softmax_arr[j:j+inputs.shape[0],:] = model(inputs).detach().cpu().numpy()
            j = j + inputs.shape[0]

    # Apply softmax to logits
    softmax_arr = softmax(softmax_arr, axis=1)

    return softmax_arr, labels_arr


def get_cal_test_softmax_and_labels(config):
    model = get_model(config)

    if config['proper_cal']:
        train_loader, val_loader, cal_loader, test_loader = get_dataloaders(config)
        print('Computing cal softmax scores...')
        cal_softmax, cal_labels = get_softmax_and_labels(cal_loader, model, config)
        np.save(cache_folder + config['model_filename'] + '_cal_softmax.npy', cal_softmax)
        np.save(cache_folder + config['model_filename'] + '_cal_labels.npy', cal_labels)
    else:
        train_loader, val_loader, test_loader = get_dataloaders(config)
        print('Computing val softmax scores...')
        val_softmax, val_labels = get_softmax_and_labels(val_loader, model, config)
        np.save(cache_folder + config['model_filename'] + '_val_softmax.npy', val_softmax)
        np.save(cache_folder + config['model_filename'] + '_val_labels.npy', val_labels)

    
    print('Computing test softmax scores...')
    test_softmax, test_labels = get_softmax_and_labels(test_loader, model, config)
    np.save(cache_folder + config['model_filename'] + '_test_softmax.npy', test_softmax)
    np.save(cache_folder + config['model_filename'] + '_test_labels.npy', test_labels)
    
    print('Saved val/cal and test softmax+labels to ' + cache_folder + config['model_filename'] 
          + '[...].npy')

    # Also save train_labels to use later
    _, _, _, train_labels, _, _ = get_datasets(config['dataset_name'], 
                                               truncate=config['truncate'], return_labels=True)
    if config['truncate']:
        dset_name = config['dataset_name'] + '-trunc'
    else:
        dset_name =  config['dataset_name']
    np.save(cache_folder + dset_name + '_train_labels.npy', train_labels)
    print('Saved train labels to ' + cache_folder + dset_name + '_train_labels.npy' )


    # Sanity check accuracies
    if config['proper_cal']:
        cal_preds = np.argmax(cal_softmax, axis=1)
        print('Cal accuracy:', np.mean(cal_preds == cal_labels))
    else:
        val_preds = np.argmax(val_softmax, axis=1)
        print('Val accuracy:', np.mean(val_preds == val_labels))
    test_preds = np.argmax(test_softmax, axis=1)
    print('Test accuracy:', np.mean(test_preds == test_labels), 
          '(Note that when using --trunc, we do not expect test and val accuracies to match.)')  

    if config['proper_cal']:
        return cal_softmax, cal_labels, test_softmax, test_labels
    
    return val_softmax, val_labels, test_softmax, test_labels
    

def postprocess_config(config):
    if config['dataset_name'] == 'plantnet' and config['truncate'] == False:
        config['num_classes'] = 1081
    elif config['dataset_name'] == 'plantnet' and config['truncate'] == True:
        config['num_classes'] = 330
    elif config['dataset_name'] == 'inaturalist' and config['truncate'] == False:
        config['num_classes'] = 8142
    elif config['dataset_name'] == 'inaturalist' and config['truncate'] == True:
        config['num_classes'] = 857
    else:
        raise NotImplementedError
    return config

def get_model(config):
    # Read in folder from folders.json
    global cache_folder
    cache_folder = get_inputs_folder()
    cache_folder = cache_folder if cache_folder.endswith('/') else cache_folder + '/'
    # For focal loss, put results in a sub-directoary
    if config['loss'] == 'focal':
        cache_folder += 'focal_loss'   
    os.makedirs(cache_folder, exist_ok=True)
        
    
    model = resnet50(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    model = model.to(config['device'])
    try:
        state_dict = torch.load(cache_folder + config['model_filename'] + '.pth', map_location=config['device'])
        model.load_state_dict(state_dict)
        model.eval()
        with open(cache_folder + config['model_filename'] + '-config.pkl', 'rb') as f:
            loaded_config = pickle.load(f)
            
        for setting in ['num_classes', 'batch_size', 'num_epochs', 
                        'use_last_epoch', 'proper_cal',
                        'lr', 'dataset_name', 'truncate']:
            assert config[setting] == loaded_config[setting] # If the configs aren't equal, retrain

    except Exception:
        print(traceback.format_exc()) # Print error for debugging purposes
        os.makedirs(cache_folder, exist_ok=True)
        model = model.to(config['device'])
        dataloaders = get_dataloaders(config)
        train_loader, val_loader = dataloaders[0], dataloaders[1]
        model, val_acc_history = train_model(model, train_loader, val_loader, config) 
    return model

    


        
        

    
    