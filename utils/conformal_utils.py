import json
import matplotlib.pyplot as plt
import numpy as np
import pickle

import pdb

from collections import Counter, defaultdict
from collections.abc import Mapping
from scipy.sparse import lil_matrix
from scipy.stats import multivariate_normal, rv_discrete

# # For clustered conformal
# from utils.clustering_utils import embed_all_classes
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


#========================================
#  Misc.
#========================================

def get_quantile_threshold(alpha):
    '''
    Compute smallest n such that ceil((n+1)*(1-alpha)/n) <= 1
    '''
    n = 1
    while np.ceil((n+1)*(1-alpha)/n) > 1:
        n += 1
    return n

#========================================
#   Data preparation
#========================================

# Used for creating randomly sampled calibration dataset
def random_split(X, y, avg_num_per_class, seed=0):
    '''
    Randomly splits X and y into X1, y1, X2, y2. X1 and y1 will have of
    avg_num_per_class * num_classes examples. The remaining examples
    will be in X2 and y2.
    
    Inputs:
    - X: numpy array. axis 0 must be the same length as y
    - y: 1-D numpy array of class labels (classes should be 0-indexed)
    - avg_num_per_class: Average number of examples per class to include 
    in the data split. 
    - seed: (int) random seed for reproducibility
    
    Output: X1, y1, X2, y2
    '''

    np.random.seed(seed)
    
    num_classes = np.max(y) + 1
    
    num_samples = avg_num_per_class * num_classes
    
    idx1 = np.random.choice(np.arange(len(y)), size=num_samples, replace=False) # Numeric index
    idx2 = ~np.isin(np.arange(len(y)), idx1) # Boolean index
    X1, y1 = X[idx1], y[idx1]
    X2, y2 = X[idx2], y[idx2]
    
    return X1, y1, X2, y2


# Used for creating balanced or stratified calibration dataset
def split_X_and_y(X, y, n_k, num_classes, seed=0, split='balanced'):
    '''
    Randomly generate two subsets of features X and corresponding labels y such that the
    first subset contains n_k instances of each class k and the second subset contains all
    other instances 
    
    Inputs:
        X: n x d array (e.g., matrix of softmax vectors)
        y: n x 1 array
        n_k: positive int or n x 1 array
        num_classes: total number of classes, corresponding to max(y)
        seed: random seed
        
    Output:
        X1, y1
        X2, y2
    '''
    np.random.seed(seed)
    
    if split == 'balanced':
    
        if not hasattr(n_k, '__iter__'):
            n_k = n_k * np.ones((num_classes,), dtype=int)
    elif split == 'proportional':
        assert not hasattr(n_k, '__iter__')
        
        # Compute what fraction of the rarest class n_clustering corresponds to,
        # then set n_k = frac * (total # of cal points for class k)
        cts = Counter(y)
        rarest_class_ct = cts.most_common()[-1][1]
        frac = n_k / rarest_class_ct
        n_k = [int(frac*cts[k]) for k in range(num_classes)]
    elif split == 'frac':
        # Split so that n_k fraction of each class is allocated to X1, y1.
        # Round so that more examples end up in X2, y2. Notably, this ensures y2 has
        # at lest one example from each class.
        # In this case, n_k should be a fraction between 0 and 1 (rather than a number)
        cts = Counter(y)
        n_k = [int(np.floor(cts[k]*n_k)) for k in range(num_classes)]
    else: 
        raise Exception('Valid split options are "balanced" or "proportional"')
            
    
    if len(X.shape) == 2:
        X1 = np.zeros((np.sum(n_k), X.shape[1]))
    else:
        X1 = np.zeros((np.sum(n_k),))
    y1 = np.zeros((np.sum(n_k), ), dtype=np.int32)
    
    all_selected_indices = np.zeros(y.shape)

    i = 0
    for k in range(num_classes):

        # Randomly select n instances of class k
        idx = np.argwhere(y==k).flatten()
        selected_idx = np.random.choice(idx, replace=False, size=(n_k[k],))

        X1[i:i+n_k[k]] = X[selected_idx]
        y1[i:i+n_k[k]] = k
        i += n_k[k]
        
        all_selected_indices[selected_idx] = 1
        
    X2 = X[all_selected_indices == 0]
    y2 = y[all_selected_indices == 0]
    
    return X1, y1, X2, y2

def get_true_class_conformal_score(scores_all, labels):
    '''
    Extracts conformal scores that corresponds to the true class labels
    
    Inputs:
        scores_all: n x num_classes array 
        labels: length-n array of true class labels
        
    Output: length-n array of scores corresponding to true label for each entry
    '''
    return scores_all[np.arange(len(labels)), labels]


def get_conformal_scores(softmax_scores, score, train_labels_path=None, weights=None):

    if score == 'softmax':
        scores = np.array(1 - softmax_scores)
    elif score == 'PAS': # Prevalence-adjusted softmax
        assert train_labels_path is not None, 'Must specify train_labels_path to use PAS score function'
        train_labels = np.load(train_labels_path)
        num_classes = train_labels.max() + 1
        train_class_distr = np.array([np.sum(train_labels == k) for k in range(num_classes)]) / len(train_labels) 
        scores = 1 - (np.array(softmax_scores) / train_class_distr)
    elif score == 'WPAS': # Weighted prevalence-adjusted softmax
        assert train_labels_path is not None, 'Must specify train_labels_path to use PAS score function'
        assert weights is not None, 'Must specify weights to use PAS score function'
        train_labels = np.load(train_labels_path)
        num_classes = train_labels.max() + 1
        train_class_distr = np.array([np.sum(train_labels == k) for k in range(num_classes)]) / len(train_labels) 
        scores = 1 - (weights * np.array(softmax_scores) / train_class_distr)
    elif score == 'APS':
        scores = np.array(get_APS_scores_all(softmax_scores, randomize=True, seed=0))
    elif score == 'RAPS':
        scores = np.array(get_RAPS_scores_all(softmax_scores, lmbda, kreg, randomize=True, seed=0))
    else:
        raise ValueError("Invalid score")

    return scores 


#========================================
#   General conformal utils
#========================================

def get_conformal_quantile(scores, alpha, default_qhat=np.inf, exact_coverage=False):
    '''
    Compute finite-sample-adjusted 1-alpha quantile of scores
    
    Inputs:
        - scores: num_instances-length array of conformal scores for true class. A higher score 
            indicates more uncertainty
        - alpha: float between 0 and 1 specifying coverage level
        - default_qhat: the value that will be returned if there are insufficient samples to compute
        the quantile. Should be np.inf if you want a coverage guarantee.
        - exact_coverage: If True return a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        such that if you use q_hat = q_a w.p. gamma and q_b w.p. 1-gamma, you achieve exact 1-alpha
        coverage
    
    '''
    if exact_coverage:
        q_a, q_b, gamma = get_exact_coverage_conformal_params(scores, alpha, default_qhat=np.inf)
        exact_cov_params = {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        return exact_cov_params
    
    else:
        n = len(scores)
        if default_qhat == 'max':
            default_qhat = np.max(scores)

        if n == 0:
            print(f'Using default q_hat of {default_qhat} because n={n}')
            return default_qhat

        val = np.ceil((n+1)*(1-alpha))/n
        if val > 1:
            print(f'Using default q_hat of {default_qhat} because n={n}')
            qhat = default_qhat
        else:
            # qhat = np.quantile(np.concatenate((scores, [np.inf])), 1-alpha, method='inverted_cdf')
            qhat = np.quantile(scores, val, method='inverted_cdf')
            # qhat = quantile(scores, val)

        return qhat

#========================================
#   Standard conformal prediction
#========================================

def compute_qhat(scores_all, true_labels, alpha, exact_coverage=False, plot_scores=False):
    '''
    Compute quantile q_hat that will result in marginal coverage of (1-alpha)
    
    Inputs:
        - scores_all: num_instances x num_classes array of scores, or num_instances-length array of 
        conformal scores for true class. A higher score indicates more uncertainty
        - true_labels: num_instances length array of ground truth labels
        - alpha: float between 0 and 1 specifying coverage level
        - plot_scores: If True, plot histogram of true class scores 
    '''
    # If necessary, select scores that correspond to correct label
    if len(scores_all.shape) == 2:
        scores = np.squeeze(np.take_along_axis(scores_all, np.expand_dims(true_labels, axis=1), axis=1))
    else:
        scores = scores_all
    
    q_hat = get_conformal_quantile(scores, alpha, exact_coverage=exact_coverage)
    
    # Plot score distribution
    if plot_scores:
        plt.hist(scores)
        plt.title('Score distribution')
        plt.show()

    return q_hat

# Create prediction sets
def create_prediction_sets(scores_all, q_hat, exact_coverage=False):
    '''
    Create standard conformal prediction sets
    
    Inputs:
        - scores_all: num_instances x num_classes array of scores
        - q_hat: conformal quantile, as returned from compute_qhat()
    '''
    if exact_coverage:
        assert isinstance(q_hat, Mapping), ('To create classwise prediction sets with exact coverage, '   
        'you must pass in q_hats computed with exact_coverage=True')
        
        q_a, q_b, gamma = q_hat['q_a'], q_hat['q_b'], q_hat['gamma']
        set_preds = construct_exact_coverage_standard_sets(q_a, q_b, gamma, scores_all)
        
    else:   
        assert(not hasattr(q_hat, '__iter__')), "q_hat should be a single number and not a list or array"
        scores_all = np.array(scores_all)
        set_preds = []
        num_samples = len(scores_all)
        for i in range(num_samples):
            set_preds.append(np.where(scores_all[i,:] <= q_hat)[0])
        
    return set_preds

# Standard conformal pipeline
def standard_conformal(cal_scores_all, cal_labels, val_scores_all, val_labels, alpha, exact_coverage=False):
    '''
    Use cal_scores_all and cal_labels to compute 1-alpha conformal quantiles for standard conformal.
    If exact_coverage is True, apply randomized to achieve exact 1-alpha coverage. Otherwise, use
    unrandomized conservative sets. 
    Create predictions and compute evaluation metrics on val_scores_all and val_labels.
    '''
    
    standard_qhat = compute_qhat(cal_scores_all, cal_labels, alpha, exact_coverage=exact_coverage)
    standard_preds = create_prediction_sets(val_scores_all, standard_qhat, exact_coverage=exact_coverage)
    coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, standard_preds, alpha)
    
    return standard_qhat, standard_preds, coverage_metrics, set_size_metrics

#========================================
#   Classwise conformal prediction
#========================================

def compute_class_specific_qhats(cal_scores_all, cal_true_labels, num_classes, alpha, 
                                 default_qhat=np.inf, null_qhat=None, regularize=False,
                                 exact_coverage=False):
    '''
    Computes class-specific quantiles (one for each class) that will result in coverage of (1-alpha)
    
    Inputs:
        - cal_scores_all: 
            num_instances x num_classes array where cal_scores_all[i,j] = score of class j for instance i
            OR
            num_instances length array where entry i is the score of the true label for instance i
        - cal_true_labels: num_instances-length array of true class labels (0-indexed). 
            [Useful for implenting clustered conformal] If class -1 appears, it will be assigned the 
            default_qhat value. It is appended as an extra entry of the returned q_hats so that q_hats[-1] = null_qhat. 
        - alpha: Determines desired coverage level
        - default_qhat: Float, or 'standard'. For classes that do not appear in cal_true_labels, the class 
        specific qhat is set to default_qhat. If default_qhat == 'standard', we compute the qhat for standard
        conformal and use that as the default value. Should be np.inf if you want a coverage guarantee.
        - null_qhat: Float, or 'standard', as described for default_qhat. Only used if -1 appears in
        cal_true_labels. null_qhat is assigned to class/cluster -1 
        - regularize: If True, shrink the class-specific qhat towards the default_qhat value. Amount of
        shrinkage for a class is determined by number of samples of that class. Only implemented for 
        exact_coverage=False.
        - exact_coverage: If True return a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        such that if you use q_hat = q_a w.p. gamma and q_b w.p. 1-gamma, you achieve exact 1-alpha
        coverage
    '''
    # Extract conformal scores for true labels if not already done
    if len(cal_scores_all.shape) == 2:
        cal_scores_all = cal_scores_all[np.arange(len(cal_true_labels)), cal_true_labels]
       
    if exact_coverage:
        if default_qhat == 'standard':
            default_qhat = get_exact_coverage_conformal_params(cal_scores_all, alpha, default_qhat=default_qhat)
        q_a, q_b, gamma = compute_exact_coverage_class_specific_params(cal_scores_all, cal_true_labels, 
                                                                       num_classes, alpha, 
                                                                       default_qhat=default_qhat, null_params=null_qhat)
        exact_cov_params = {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        
        return exact_cov_params
    
    else:
    
        if default_qhat == 'standard':
            default_qhat = compute_qhat(cal_scores_all, cal_true_labels, alpha=alpha)

        # If we are regularizing the qhats, we should set aside some data to correct the regularized qhats 
        # to get a marginal coverage guarantee
        if regularize:
            num_reserve = min(1000, int(.5*len(cal_true_labels))) # Reserve 1000 or half of calibration set, whichever is smaller
            idx = np.random.choice(np.arange(len(cal_true_labels)), size=num_reserve, replace=False)
            idx2 = ~np.isin(np.arange(len(cal_true_labels)), idx)
            reserved_scores, reserved_labels = cal_scores_all[idx], cal_true_labels[idx]
            cal_scores_all, cal_true_labels = cal_scores_all[idx2], cal_true_labels[idx2]


        q_hats = np.zeros((num_classes,)) # q_hats[i] = quantile for class i
        class_cts = np.zeros((num_classes,))

        for k in range(num_classes):

            # Only select data for which k is true class
            idx = (cal_true_labels == k)
            scores = cal_scores_all[idx]

            class_cts[k] = scores.shape[0]

            q_hats[k] = get_conformal_quantile(scores, alpha, default_qhat=default_qhat)
            
            
        if -1 in cal_true_labels:
            q_hats = np.concatenate((q_hats, [null_qhat]))


        # Optionally apply shrinkage 
        if regularize:
            N = num_classes
            n_k = np.maximum(class_cts, 1) # So that classes that never appear do not cause division by 0 issues. 
            shrinkage_factor = .03 * n_k # smaller = less shrinkage
            shrinkage_factor = np.minimum(shrinkage_factor, 1)
            print('SHRINKAGE FACTOR:', shrinkage_factor)  
            print(np.min(shrinkage_factor), np.max(shrinkage_factor))
            q_hats = default_qhat + shrinkage_factor * (q_hats - default_qhat)

            # Correct qhats via additive factor to achieve marginal coverage
            q_hats = reconformalize(q_hats, reserved_scores, reserved_labels, alpha)


        return q_hats

# Create classwise prediction sets
def create_classwise_prediction_sets(scores_all, q_hats, exact_coverage=False):
    '''
    Inputs:
        - scores_all: num_instances x num_classes array where scores_all[i,j] = score of class j for instance i
        - q_hats: as output by compute_class_specific_quantiles
        - exact_coverage: Must match the exact_coverage setting used to compute q_hats. 
    '''
    if exact_coverage:
        assert isinstance(q_hats, Mapping), ('To create classwise prediction sets with exact coverage, '   
        'you must pass in q_hats computed with exact_coverage=True')
        
        q_as, q_bs, gammas = q_hats['q_a'], q_hats['q_b'], q_hats['gamma']
        set_preds = construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, scores_all)
    
    else:
        scores_all = np.array(scores_all)
        set_preds = []
        num_samples = len(scores_all)
        for i in range(num_samples):
            set_preds.append(np.where(scores_all[i,:] <= q_hats)[0])

    return set_preds


# Classwise conformal pipeline
def classwise_conformal(totalcal_scores_all, totalcal_labels, val_scores_all, val_labels, alpha,
                         num_classes, default_qhat=np.inf, regularize=False, exact_coverage=False):
    '''
    Use cal_scores_all and cal_labels to compute 1-alpha conformal quantiles for classwise conformal.
    If exact_coverage is True, apply randomized to achieve exact 1-alpha coverage. Otherwise, use
    unrandomized conservative sets. 
    Create predictions and compute evaluation metrics on val_scores_all and val_labels.
    
    See compute_class_specific_qhats() docstring for more details about expected inputs.
    '''
    
    classwise_qhats = compute_class_specific_qhats(totalcal_scores_all, totalcal_labels, 
                                                   alpha=alpha, 
                                                   num_classes=num_classes,
                                                   default_qhat=default_qhat, regularize=regularize,
                                                   exact_coverage=exact_coverage)
    classwise_preds = create_classwise_prediction_sets(val_scores_all, classwise_qhats, exact_coverage=exact_coverage)

    coverage_metrics, set_size_metrics = compute_all_metrics(val_labels, classwise_preds, alpha)
    
    return classwise_qhats, classwise_preds, coverage_metrics, set_size_metrics
                

#========================================
#   Clustered conformal prediction
#========================================

def quantile_embedding(samples, q=[0.5, 0.6, 0.7, 0.8, 0.9]):
    '''
    Computes the q-quantiles of samples and returns the vector of quantiles
    '''
    return np.quantile(samples, q)

def embed_all_classes(scores_all, labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=False):
    '''
    Input:
        - scores_all: num_instances x num_classes array where 
            scores_all[i,j] = score of class j for instance i
          Alternatively, num_instances-length array where scores_all[i] = score of true class for instance i
        - labels: num_instances-length array of true class labels
        - q: quantiles to include in embedding
        - return_cts: if True, return an array containing the counts for each class 
        
    Output: 
        - embeddings: num_classes x len(q) array where ith row is the embeddings of class i
        - (Optional) cts: num_classes-length array where cts[i] = # of times class i 
        appears in labels 
    '''
    num_classes = len(np.unique(labels))
    
    embeddings = np.zeros((num_classes, len(q)))
    cts = np.zeros((num_classes,))
    
    for i in range(num_classes):
        if len(scores_all.shape) == 2:
            class_i_scores = scores_all[labels==i,i]
        else:
            class_i_scores = scores_all[labels==i] 
        cts[i] = class_i_scores.shape[0]
        embeddings[i,:] = quantile_embedding(class_i_scores, q=q)
    
    if return_cts:
        return embeddings, cts
    else:
        return embeddings

def compute_cluster_specific_qhats(cluster_assignments, cal_scores_all, cal_true_labels, alpha, 
                                   null_qhat='standard', exact_coverage=False):
    '''
    Computes cluster-specific quantiles (one for each class) that will result in coverage of (1-alpha)
    
    Inputs:
        - cluster_assignments: num_classes length array where entry i is the index of the cluster that class i belongs to.
          Clusters should be 0-indexed. Rare classes can be assigned to cluster -1 and they will automatically be given
          qhat_k = default_qhat. 
        - cal_scores_all: num_instances x num_classes array where scores_all[i,j] = score of class j for instance i.
         Alternatively, a num_instances-length array of conformal scores for true class
        - cal_true_labels: num_instances length array of true class labels (0-indexed)
        - alpha: Determines desired coverage level
        - null_qhat: For classes that do not appear in cal_true_labels, the class specific qhat is set to null_qhat.
        If null_qhat == 'standard', we compute the qhat for standard conformal and use that as the default value
        - exact_coverage: If True, return a dict of values {'q_a': q_a, 'q_b': q_b, 'gamma': gamma}
        such that if you use q_hat = q_a w.p. gamma and q_b w.p. 1-gamma, you achieve exact 1-alpha
        coverage
         
    Output:
        num_classes length array where entry i is the quantile corresponding to the cluster that class i belongs to. 
        All classes in the same cluster have the same quantile.
        
        OR (if exact_coverage=True), dict containing clustered conformal parameters needed to achieve exact coverage
    '''
    # If we want the null_qhat to be the standard qhat, we should compute this using the original class labels
    if null_qhat == 'standard' and not exact_coverage:
        null_qhat = compute_qhat(cal_scores_all, cal_true_labels, alpha)
            
    # Extract conformal scores for true labels if not already done
    if len(cal_scores_all.shape) == 2:
        cal_scores_all = cal_scores_all[np.arange(len(cal_true_labels)), cal_true_labels]
        
    # Edge case: all cluster_assignments are -1. 
    if np.all(cluster_assignments==-1):
        if exact_coverage:
            null_qa, null_qb, null_gamma = get_exact_coverage_conformal_params(cal_scores_all, alpha) # Assign standard conformal params to null cluster
            q_as = null_qa * np.ones(cluster_assignments.shape)
            q_bs = null_qb * np.ones(cluster_assignments.shape)
            gammas = null_gamma * np.ones(cluster_assignments.shape)
            return q_as, q_bs, gammas
        else:
            return null_qhat * np.ones(cluster_assignments.shape)
    
    # Map true class labels to clusters
    cal_true_clusters = np.array([cluster_assignments[label] for label in cal_true_labels])
    
    # Compute cluster qhats
    if exact_coverage:
        if null_qhat == 'standard':
            null_qa, null_qb, null_gamma = get_exact_coverage_conformal_params(cal_scores_all, alpha) # Assign standard conforml params to null cluster
            null_params = {'q_a': null_qa, 'q_b': null_qb, 'gamma': null_gamma}
        clustq_as, clustq_bs, clustgammas = compute_exact_coverage_class_specific_params(cal_scores_all, cal_true_clusters,
                                                                          num_classes=np.max(cluster_assignments)+1, 
                                                                          alpha=alpha, 
                                                                          default_qhat=np.inf, null_params=null_params)
        # Map cluster qhats back to classes
        num_classes = len(cluster_assignments)
        q_as = np.array([clustq_as[cluster_assignments[k]] for k in range(num_classes)])
        q_bs = np.array([clustq_bs[cluster_assignments[k]] for k in range(num_classes)])
        gammas = np.array([clustgammas[cluster_assignments[k]] for k in range(num_classes)])

        return q_as, q_bs, gammas
   
    else:
            
        cluster_qhats = compute_class_specific_qhats(cal_scores_all, cal_true_clusters, 
                                                     alpha=alpha, num_classes=np.max(cluster_assignments)+1,
                                                     default_qhat=np.inf,
                                                     null_qhat=null_qhat)                            
        # Map cluster qhats back to classes
        num_classes = len(cluster_assignments)
        class_qhats = np.array([cluster_qhats[cluster_assignments[k]] for k in range(num_classes)])

        return class_qhats

    # Note: To create prediction sets, just pass class_qhats into create_classwise_prediction_sets()
    
    
def clustered_conformal(totalcal_scores_all, totalcal_labels,
                        alpha,
                        val_scores_all=None, val_labels=None,
                        frac_clustering='auto', num_clusters='auto',
                        split='random',
                        exact_coverage=False, seed=0):
    '''
    Use totalcal_scores and totalcal_labels to compute conformal quantiles for each
    class using the clustered conformal procedure. Optionally evaluates 
    performance on val_scores and val_labels
    
    Clustered conformal procedure:
        1. Filter out classes where np.ceil((num_samples+1)*(1-alpha)) / num_samples < 1. 
           Assign those classes the standard qhat
        2. To select n_clustering (or frac_clustering) and num_clusters, apply heuristic 
           (which takes as input num_samples and num_classes) to num_samples of rarest remaining class
        3. Run clustering with chosen hyperparameters. Estimate a qhat for each cluster.
    
    Inputs:
         - totalcal_scores: num_instances x num_classes array where 
           cal_scores_all[i,j] = score of class j for instance i
         - totalcal_labels: num_instances-length array of true class labels (0-indexed classes)
         - alpha: number between 0 and 1 that determines coverage level.
         Coverage level will be 1-alpha.
         - n_clustering: Number of points per class to use for clustering step. The remaining
         points are used for the conformal calibration step.
         - num_clusters: Number of clusters to group classes into
         - val_scores: num_val_instances x num_classes array, or None. If not None, 
         the class coverage gap and average set sizes will be computed on val_scores
         and val_labels.
         - val_labels: num_val_instances-length array of true class labels, or None. 
         If not None, the class coverage gap and average set sizes will be computed 
         on val_scores and val_labels.
         - split: How to split data between clustering step and calibration step. Options are
         'balanced' (sample n_clustering per class), 'proportional' (sample proportional to
         distribution such that rarest class has n_clustering example), 'doubledip' (don't
         split and use all data for both steps, or 'random' (each example is assigned to
         clustering step with some fixed probability) 
         - exact_coverage: If True, randomize the prediction sets to achieve exact 1-alpha coverage. Otherwise, 
         the sets will be unrandomized but slightly conservative
         - seed: Random seed. Affects split between clustering set and proper calibraton set
         
    Outputs:
        - qhats: num_classes-length array where qhats[i] = conformal quantial estimate for class i
        - [Optionally, if val_scores and val_labels are not None] 
            - val_preds: clustered conformal predictions on val_scores
            - val_class_coverage_gap: Class coverage gap, compute on val_scores and val_labels
            - val_set_size_metrics: Dict containing set size metrics, compute on val_scores and val_labels
            
    '''
    np.random.seed(seed) 
    
    def get_rare_classes(labels, alpha, num_classes):
        thresh = get_quantile_threshold(alpha)
        classes, cts = np.unique(labels, return_counts=True)
        rare_classes = classes[cts < thresh]
        
        # Also included any classes that are so rare that we have 0 labels for it
        zero_ct_classes = np.setdiff1d(np.arange(num_classes), classes)
        rare_classes = np.concatenate((rare_classes, zero_ct_classes))
        
        return rare_classes
        
    def remap_classes(labels, rare_classes):
        '''
        Exclude classes in rare_classes and remap remaining classes to be 0-indexed

        Outputs:
            - remaining_idx: Boolean array the same length as labels. Entry i is True
            iff labels[i] is not in rare_classes 
            - remapped_labels: Array that only contains the entries of labels that are 
            not in rare_classes (in order) 
            - remapping: Dict mapping old class index to new class index

        '''
        remaining_idx = ~np.isin(labels, rare_classes)

        remaining_labels = labels[remaining_idx]
        remapped_labels = np.zeros(remaining_labels.shape, dtype=int)
        new_idx = 0
        remapping = {}
        for i in range(len(remaining_labels)):
            if remaining_labels[i] in remapping:
                remapped_labels[i] = remapping[remaining_labels[i]]
            else:
                remapped_labels[i] = new_idx
                remapping[remaining_labels[i]] = new_idx
                new_idx += 1
        return remaining_idx, remapped_labels, remapping
    
    # Data preperation: Get conformal scores for true classes
    num_classes = totalcal_scores_all.shape[1]
    totalcal_scores = get_true_class_conformal_score(totalcal_scores_all, totalcal_labels)
    
    # 1) Apply heuristic to choose hyperparameters if not prespecified
    if frac_clustering == 'auto' and num_clusters == 'auto':
        cts_dict = Counter(totalcal_labels)
        cts = [cts_dict.get(k, 0) for k in range(num_classes)]
        n_min = min(cts)
        n_thresh = get_quantile_threshold(alpha) 
        n_min = max(n_min, n_thresh) # Classes with fewer than n_thresh examples will be excluded from clustering
        num_remaining_classes = np.sum(np.array(list(cts)) >= n_min)

        n_clustering, num_clusters = get_clustering_parameters(num_remaining_classes, n_min)
        print(f'n_clustering={n_clustering}, num_clusters={num_clusters}')
        # Convert n_clustering to fraction relative to n_min
        frac_clustering = n_clustering / n_min

        
    # 2a) Split data
    if split == 'proportional':
        n_k = [int(frac_clustering*cts[k]) for k in range(num_classes)]
        scores1, labels1, scores2, labels2 = split_X_and_y(totalcal_scores, 
                                                           totalcal_labels, 
                                                           n_k, 
                                                           num_classes=num_classes, 
                                                           seed=0)
#                                                            split=split, # Balanced or stratified sampling 
    elif split == 'doubledip':
        scores1, labels1 = totalcal_scores, totalcal_labels
        scores2, labels2 = totalcal_scores, totalcal_labels
    elif split == 'random':
        # Each point is assigned to clustering set w.p. frac_clustering 
        idx1 = np.random.uniform(size=(len(totalcal_labels),)) < frac_clustering 
        scores1 = totalcal_scores[idx1]
        labels1 = totalcal_labels[idx1]
        scores2 = totalcal_scores[~idx1]
        labels2 = totalcal_labels[~idx1]
        
    else:
        raise Exception('Invalid split. Options are balanced, proportional, doubledip, and random')

    # 2b)  Identify "rare" classes = classes that have fewer than 1/alpha - 1 examples 
    # in the clustering set 
    rare_classes = get_rare_classes(labels1, alpha, num_classes)
    print(f'{len(rare_classes)} of {num_classes} classes are rare in the clustering set'
          ' and will be assigned to the null cluster')
    
    # 3) Run clustering
    if num_classes - len(rare_classes) > num_clusters and num_clusters > 1:  
        # Filter out rare classes and re-index
        remaining_idx, filtered_labels, class_remapping = remap_classes(labels1, rare_classes)
        filtered_scores = scores1[remaining_idx]
        
        # Compute embedding for each class and get class counts
        embeddings, class_cts = embed_all_classes(filtered_scores, filtered_labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=True)
    
        kmeans = KMeans(n_clusters=int(num_clusters), random_state=0, n_init=10).fit(embeddings, sample_weight=np.sqrt(class_cts))
        nonrare_class_cluster_assignments = kmeans.labels_  

        # Print cluster sizes
        print(f'Cluster sizes:', [x[1] for x in Counter(nonrare_class_cluster_assignments).most_common()])

        # Remap cluster assignments to original classes. Any class not included in kmeans clustering is a rare 
        # class, so we will assign it to cluster "-1" = num_clusters by Python indexing
        cluster_assignments = -np.ones((num_classes,), dtype=int)
        for cls, remapped_cls in class_remapping.items():
            cluster_assignments[cls] = nonrare_class_cluster_assignments[remapped_cls]
    else: 
        cluster_assignments = -np.ones((num_classes,), dtype=int)
        print('Skipped clustering because the number of clusters requested was <= 1')
        
    # 4) Compute qhats for each cluster
    cal_scores_all = scores2
    cal_labels = labels2
    if exact_coverage: 
        q_as, q_bs, gammas = compute_cluster_specific_qhats(cluster_assignments, cal_scores_all, cal_labels, alpha, 
                                   null_qhat='standard', exact_coverage=True)
        
    else: 
        qhats = compute_cluster_specific_qhats(cluster_assignments, 
                   cal_scores_all, cal_labels, 
                   alpha=alpha, 
                   null_qhat='standard')
        

    # 5) [Optionally] Apply to val set. Evaluate class coverage gap and set size 
    if (val_scores_all is not None) and (val_labels is not None):
        if exact_coverage:
            preds = construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, val_scores_all)
            qhats = {'q_a': q_as, 'q_b': q_bs, 'gamma': gammas} # Alias for function return
        else:
            preds = create_classwise_prediction_sets(val_scores_all, qhats)
        class_cov_metrics, set_size_metrics = compute_all_metrics(val_labels, preds, alpha,
                                                                  cluster_assignments=cluster_assignments)

        # Add # of classes excluded from clustering to class_cov_metrics
        class_cov_metrics['num_unclustered_classes'] = len(rare_classes)
        
        return qhats, preds, class_cov_metrics, set_size_metrics
    else:
        return qhats

    
def get_clustering_parameters(num_classes, n_totalcal):
    '''
    Returns a guess of good values for num_clusters and n_clustering based solely 
    on the number of classes and the number of examples per class. 
    
    This relies on two heuristics:
    1) We want at least 150 points per cluster on average
    2) We need more samples as we try to distinguish between more distributions. 
    To distinguish between 2 distribution, want at least 4 samples per class. 
    To distinguish between 5 distributions, want at least 10 samples per class. 
    
    Output: n_clustering, num_clusters
    
    '''
    # Alias for convenience
    K = num_classes
    N = n_totalcal
    
    n_clustering = int(N*K/(75+K))
    num_clusters = int(np.floor(n_clustering / 2))
    
    return n_clustering, num_clusters

#========================================
#   Fuzzy Classwise CP
#========================================

def fuzzy_classwise_CP(cal_scores_all, cal_labels, alpha, val_scores_all=None, 
                       projection='HRS', mode='weight', params={}, 
                       show_diagnostics=False, reconformalize=None,
                       reconformalize_data=None):
    '''
    params: 
    * When mode='bin', params['n_bins'] must be defined. 
    * When mode='weight', 
        params['bandwidth'] must be defined. 
        params['kernel'] can be specified. Default is 'gaussian'. 
        Other option is 'dirac_uniform'. In that case, 'bandwidth' in [0,1] specifies mass to put on 
    * reconformalize: Options are None or "additive"
    * reconformalize_data: If not None, should be of form (holdout_scores_all, holdout_labels)
    * [No longer used] If params['FPR_tol'] is defined, class_qhats will be post-processed to "give up" on classes that
      yield false positive rates above the provided tolerance. The qhat will be replaced with params['default_qhat']
      if it is specified, otherwise 0. 

    '''
    np.random.seed(1) # for reproducibility if using a projection with randomness

    # 0) Data prep
    num_classes = cal_scores_all.shape[1]
    # Select scores that correspond to correct label
    cal_scores = np.squeeze(np.take_along_axis(cal_scores_all, np.expand_dims(cal_labels, axis=1), axis=1))

    # 1) Get projection of each class
    if projection == 'rarity':
        if params.get('use_train', False):
            labels_path = params.get('train_labels_path', f'train_models/.cache/{params["dataset"]}_train_labels.npy')
            labels = np.load(labels_path)
            print('Computing rarity using train labels')
        else:
            labels = cal_labels
        cts = np.array([np.sum(labels == k) for k in range(num_classes)])
        
        proj_arr = cts/np.max(cts) # Normalize to [0,1]
        # Unless it is specified to not jitter, add a small amount of jitter
        # to ensure that each class is mapped to a unique value
        if params.get('jitter', True):
            proj_arr = proj_arr + np.random.uniform(low=-.01, high=.01, size=cts.shape)
    elif projection == 'random': # Project each class to a random value in [0,1]
        
        proj_arr = np.random.rand(num_classes).astype(np.float64)

    elif projection == 'quantile': # Project each class to an estimate of its 1-alpha score quantile
        # We use linear interpolation. e.g., the 0.9 quantile of [1, 2] is 1.9
        # For classes with 0 examples, we arbitrarily choose to map it to the 
        # largest score observed in the calibration data (implicitly assuming
        # that those rare classes are "hard" for the model)
        cal_scores = get_true_class_conformal_score(cal_scores_all, cal_labels)
        max_score = np.max(cal_scores)
        proj_arr = np.zeros((num_classes,))
        for k in range(num_classes):
            if np.sum(cal_labels == k) > 0:
                proj_arr[k] = np.quantile(cal_scores[cal_labels == k], 1-alpha, method='linear') 
            else:
                proj_arr[k] = max_score

    # 2) Compute class-specific quantiles
    # OPTION 1: Cluster by binning HRS
    if mode == 'bin':
        assert 'n_bins' in params, "If mode='bin', you must specify params['n_bins']"

        # 2) Define clustering based on HRS bins
        n_bins = params['n_bins']
        # Get bin edges
        quantile_bin_edges = np.quantile(proj_arr, np.linspace(0, 1, n_bins))
        # Assign each value in proj_arr to a quantile bin (0-indexed)
        cluster_assignments = np.digitize(proj_arr, quantile_bin_edges, right=False) - 1
        
        if show_diagnostics:
            # Plot number of classes per cluster
            plt.hist(cluster_assignments, bins=np.arange(max(cluster_assignments))-0.5)
            plt.xticks(np.arange(max(cluster_assignments)))
            plt.xlabel('Cluster')
            plt.ylabel('Number of classes in cluster')
            plt.show()
    
            # Plot number of data points per cluster
            data_per_cluster = np.zeros((n_bins,))
            for k in range(num_classes):
                data_per_cluster[cluster_assignments[k]] += np.sum(cal_labels == k)
            plt.bar(np.arange(n_bins), data_per_cluster)
            plt.xticks(np.arange(max(cluster_assignments)))
            plt.xlabel('Cluster')
            plt.ylabel('# of examples in cluster')
            plt.yscale('log')
            plt.title(f'Min # of examples in a cluster is {int(min(data_per_cluster))}')
            plt.show()
    
        # Compute cluster-specific qhats
        class_qhats = compute_cluster_specific_qhats(cluster_assignments, cal_scores_all, cal_labels, alpha)

    # OPTION 2: Assign weights for scores from each class based on distance from current class in HRS space
    elif mode == 'weight':
        assert 'bandwidth' in params, "If mode='weight', you must specify params['bandwidth']"
        bandwidth = params['bandwidth']

        class_qhats = np.zeros((num_classes,))
        
        if show_diagnostics:
            plt.figure(figsize=(15, 5))

        F = {} # used for "alpha" reconformalization. F[k] is the k-weighted score distribution
        for k in range(num_classes):
            n_k = np.sum(cal_labels == k)
            sd = bandwidth/(n_k+1) # + 1 to account for classes with 0 examples
            cls_weights = multivariate_normal.pdf(proj_arr, mean=proj_arr[k], cov=sd)
    
            if show_diagnostics:
                # Plot weights assigned to each class
                x = np.linspace(0,1,1000)
                normal_density = multivariate_normal.pdf(x, mean=proj_arr[k], cov=sd)
                plt.plot(x, normal_density, color='black', alpha=0.25)
                plt.scatter(proj_arr, cls_weights)
                plt.xlabel('Hierarchical rarity score')
                plt.ylabel('Un-normalized weight')
                # plt.title(f'Class {k} with {n_k} examples , using N({proj_arr[k]:.3f}, {sd:.6f})')
                # plt.show()

            # Map each calibration example to a weight depending on its class
            weights = [cls_weights[y] for y in cal_labels]

            # Add weight for test y (essentially the weighted CP version of the 1/(n+1) adjustment)
            weights += [cls_weights[k]]
            weights = np.array(weights)
            
            # Compute weighted quantile (note that due to numerical imprecision,
            # it is better to let np.quantile do the normalization of weights within the function)
            S_all = np.concatenate((cal_scores, [np.inf]))
            weighted_quantile = np.quantile(S_all, q=1 - alpha, weights=weights, method="inverted_cdf")
            
            class_qhats[k] = weighted_quantile

            # Store k-weighted score distribution
            def deduplicate(arr, weights):
                # 1.  Get the unique values and, for every original position, the index of its unique bucket
                unique_arr, inverse = np.unique(arr, return_inverse=True)
                
                # 2.  Allocate an array to hold the summed weights, one slot per unique value
                agg_weights = np.zeros_like(unique_arr, dtype=weights.dtype)
                
                # 3.  In-place scatter-add: add each weight to the right bucket
                np.add.at(agg_weights, inverse, weights)
            
                return unique_arr, agg_weights

            weights = weights / np.sum(weights)
            unique_S, agg_weights = deduplicate(S_all, weights)
            F[k] = rv_discrete(values=(unique_S, agg_weights)) 
        
        if show_diagnostics:
            plt.show()


    num_inf_qhats = np.sum(np.isinf(class_qhats))
    if num_inf_qhats > 0:
        print(f'FUZZY CP WARNING: {num_inf_qhats} classes have qhat=infinity (and will always be included in the prediction set)' )

    # 3) Re-conformalize the qhats to ensure marginal coverage by adding an
    #    additive offset
    # Pre-process: replace infinite qhats with a finite value (e.g., the max score)
    if reconformalize is not None:
        class_qhats = np.minimum(np.max(cal_scores_all), class_qhats)

        if reconformalize_data is not None: # use heldout data
            holdout_scores_all, holdout_labels = reconformalize_data
            reconf_scores = get_true_class_conformal_score(holdout_scores_all, holdout_labels) 
            reconf_labels = holdout_labels
        else: # reuse calibration data (no marginal guarantee!)
            reconf_scores = cal_scores
            reconf_labels = cal_labels

        if reconformalize == 'additive':
            S = reconf_scores - np.array([class_qhats[y] for y in reconf_labels])
            conformal_offset = get_conformal_quantile(S, alpha)
            class_qhats += conformal_offset   
        elif reconformalize == 'multiplicative':
            S = reconf_scores / np.array([class_qhats[y] for y in reconf_labels])
            conformal_offset = get_conformal_quantile(S, alpha)     
            class_qhats *= conformal_offset
        elif reconformalize == 'alpha': 
            # Find the alpha' such that taking the 1-alpha' quantile of the class-weighted score distribution
            # yields 1-alpha marginal coverage
            # This implies the score function s(x,y) = What quantile of the w(y)-weighted score distribution should I have
            # taken to include y in the prediction set?

            S = np.array([F[y].cdf(reconf_scores[i]) for i,y in enumerate(reconf_labels)]) 
            alpha_hat = 1-get_conformal_quantile(S, alpha) 
            # class_qhats = np.array([F[y].ppf(1-alpha_hat) for y in range(num_classes)])
            class_qhats = np.array([F[y].ppf(1-alpha_hat + 1e-10) for y in range(num_classes)])  # ADDED to enforce score < q instead of <=
        else:
            raise ValueError('Invalid "reconformalize" option. Options are "additive", "multiplicative", and "alpha"')
        
    # 4) [Optional] If val_scores_all is not None, construct prediction sets for it
    if val_scores_all is not None:
        preds = create_classwise_prediction_sets(np.array(val_scores_all), class_qhats)

        return class_qhats, preds, proj_arr 

    return class_qhats, proj_arr


#========================================
#   rc3p (from Shi et al., 2024)
#========================================

def compute_ranks(softmax_scores):
    '''
    Compute ranks based on softmax scores.
    Rank 0 = highest softmax probability

    Args:
        softmax_scores: (n, K) array of softmax probabilities

    Returns:
        ranks: (n, K) array of ranks
    '''
    softmax_scores = np.array(softmax_scores)
    n, K = softmax_scores.shape

    # Compute ranks for all (i, j): rank 1 = highest softmax probability
    sorted_idx = np.argsort(-softmax_scores, axis=1)        # (n, K)
    ranks = np.empty_like(sorted_idx)
    ranks[np.arange(n)[:, None], sorted_idx] = np.arange(0, K)[None, :]

    return ranks



# Compute rc3p parameters
def compute_rc3p_params(cal_softmax_scores,
                        cal_scores_all,
                        cal_labels,
                        alpha,
                        default_qhat=np.inf):
    """
    Compute rc3p per-class parameters (k_hat and q_hats) using the base conformal
    scores cal_scores_all and softmax probabilities cal_softmax_scores.

    rc3p uses:
        - a per-class rank budget k_hat[k]
        - a per-class effective miscoverage alpha_hat[k]
        - a per-class quantile q_hat_rc3p[k] computed at level 1 - alpha_hat[k]
          over the *true-class* scores of class k.

    Args:
        cal_softmax_scores: (n_cal, K) array of softmax probabilities on calibration data
        cal_scores_all:     (n_cal, K) array of base scores (APS/RAPS/etc.)
        cal_labels:         (n_cal,) array of integer class labels
        alpha:              global miscoverage level (e.g., 0.1)
        default_qhat:       default threshold when a class has no data

    Returns:
        q_hats_rc3p: (K,) array of per-class thresholds for rc3p
        k_hats:      (K,) array of per-class integer rank caps (1-based)
        alpha_hats:  (K,) array of per-class effective miscoverage (for debugging)
    """
    cal_softmax_scores = np.array(cal_softmax_scores)
    cal_scores_all = np.array(cal_scores_all)
    cal_labels = np.array(cal_labels)

    n_cal, num_classes = cal_softmax_scores.shape

    # Convert softmax score matrix into ranks (0-indexed)
    ranks = compute_ranks(cal_softmax_scores)       # (n_cal, K)


    q_hats_rc3p = np.zeros((num_classes,))
    k_hats = np.zeros((num_classes,), dtype=int)
    alpha_hats = np.zeros((num_classes,))

    for k in range(num_classes):
        # Select calibration examples where class k is the true label
        idx_k = (cal_labels == k)
        if not np.any(idx_k):
            # Default behavior for classes with no calibration examples
            q_hats_rc3p[k] = default_qhat
            k_hats[k] = num_classes  # no rank restriction
            alpha_hats[k] = alpha
            print(f'rc3p WARNING: class {k} has no calibration examples; using default qhat={default_qhat} and k_hat={num_classes}')
            continue

        ranks_k = ranks[idx_k, k]               # ranks of true class k
        scores_k = cal_scores_all[idx_k, k]     # scores of true class k

        # Use Option II from Shi et al., 2024 to select k_hat: smallest t in 1..num_classes
        # such that top-t error <= alpha. ranks_k is 0-indexed (0 = top-1), so
        # top-t error = P(rank >= t) when using 0-indexed ranks.
        k_hat = num_classes
        for t in range(1, num_classes + 1):
            top_t_error = np.mean(ranks_k > t)
            if top_t_error <= alpha:
                k_hat = t
            else:
                break

        # Sanity check: ensure the chosen k_hat meets the required bound
        if not np.mean(ranks_k > k_hat) <= alpha:
            pdb.set_trace()
        assert np.mean(ranks_k > k_hat) <= alpha, "k_hat should satisfy top-k_hat error <= alpha"
       
        alpha_hat = alpha - np.mean(ranks_k > k_hat)
        assert alpha_hat >= 0, "alpha_hat_k should be non-negative"

        qhat = get_conformal_quantile(
            scores_k,
            alpha_hat,
            default_qhat=default_qhat
        )

        k_hats[k] = k_hat
        alpha_hats[k] = alpha_hat
        q_hats_rc3p[k] = qhat

        # Estimate epsilon_{k,k} = P(rank > k | Y = k) for k=1..K
        # # eps_vec[t-1] = epsilon_t for t in 1..K
        # eps_vec = np.zeros((num_classes,))
        # for t in range(1, num_classes + 1):
        #     eps_vec[t - 1] = np.mean(ranks_k > t)

        # # Option II from the paper:
        # # k_hat[k] = min { t : epsilon_t < alpha }
        # # alpha_hat[k] = alpha - epsilon_{k_hat[k]}
        # candidates = np.where(eps_vec < alpha)[0]
        # if len(candidates) == 0:
        #     # If we never get epsilon_t < alpha, fall back to K
        #     t_idx = num_classes - 1
        # else:
        #     t_idx = candidates[0]
        # k_hat_k = t_idx + 1  # back to 1-based
        # eps_k_hat = eps_vec[t_idx]
        # alpha_hat_k = max(0.0, alpha - eps_k_hat)

        # k_hats[k] = k_hat_k
        # alpha_hats[k] = alpha_hat_k

        # # Compute class-wise quantile at level 1 - alpha_hat_k on scores_k
        # # using your existing conformal quantile helper
        # if alpha_hat_k == 0.0:
        #     # effectively require "score <= max(score)" → include everything by score;
        #     # rank truncation will still cut the set down
        #     q_hats_rc3p[k] = np.inf
        # else:
        #     q_hats_rc3p[k] = get_conformal_quantile(
        #         scores_k,
        #         alpha_hat_k,
        #         default_qhat=default_qhat,
        #         exact_coverage=False
            # )

    return q_hats_rc3p, k_hats, alpha_hats

def create_rc3p_prediction_sets(softmax_scores,
                                scores_all,
                                q_hats_rc3p,
                                k_hats):
    """
    Construct rc3p prediction sets given:
        - softmax_scores: (n, K)
        - scores_all:     (n, K) base conformal scores for all classes
        - q_hats_rc3p:    (K,) per-class thresholds from compute_rc3p_params
        - k_hats:         (K,) per-class rank caps (1-based)

    Returns:
        set_preds: list of length-n arrays, where set_preds[i] is the prediction set for example i.
    """
    softmax_scores = np.array(softmax_scores)
    scores_all = np.array(scores_all)

    n, num_classes = softmax_scores.shape

    # Get ranks
    ranks = compute_ranks(softmax_scores)       # (n, K)

    set_preds = []
    for i in range(n):
        row_scores = scores_all[i]   # shape (K,)
        row_ranks = ranks[i]         # shape (K,)

        # Include class j iff:
        #   score_ij <= q_hat_rc3p[j] and rank_ij <= k_hat[j]
        mask = (row_scores <= q_hats_rc3p) & (row_ranks <= k_hats)
        set_preds.append(np.where(mask)[0])

    return set_preds


# Full rc3p pipeline
def rc3p(cal_softmax_scores,
        cal_scores_all,
        cal_labels,
        test_softmax_scores,
        test_scores_all,
        test_labels,
        alpha,
        default_qhat=np.inf):
    """
    rc3p pipeline:
        1) Fit rc3p params (q_hats_rc3p, k_hats) on calibration data.
        2) Build prediction sets on validation data.
        3) Evaluate metrics using existing compute_all_metrics().
    """
    num_classes = cal_softmax_scores.shape[1]

    q_hats_rc3p, k_hats, alpha_hats = compute_rc3p_params(
        cal_softmax_scores,
        cal_scores_all,
        cal_labels,
        alpha,
        default_qhat
    )

    # TEMP: Sanity check: compute marginal coverage on calibration set
    cal_preds = create_rc3p_prediction_sets(
        cal_softmax_scores,
        cal_scores_all,
        q_hats_rc3p,
        k_hats
    )
    cal_coverage_metrics, _ = compute_all_metrics(cal_labels, cal_preds, alpha)
    print(f'******** rc3p calibration set marginal coverage: {cal_coverage_metrics["marginal_coverage"]:.3f}')
    

    rc3p_preds = create_rc3p_prediction_sets(
        test_softmax_scores,
        test_scores_all,
        q_hats_rc3p,
        k_hats
    )

    coverage_metrics, set_size_metrics = compute_all_metrics(test_labels, rc3p_preds, alpha)

    return (q_hats_rc3p, k_hats, alpha_hats), rc3p_preds, coverage_metrics, set_size_metrics

#========================================
#   Conformal variant: exact coverage via randomization 
#========================================

def get_exact_coverage_conformal_params(scores, alpha, default_qhat=np.inf):
    '''
    Compute the quantities necessary for performing conformal prediction with exact coverage
    
    Inputs:
        score: length-n array of conformal scores
        alpha: float between 0 and 1 denoting the desired coverage level
        default_qhat = value used when n is too small for computing the relevant quantiles
    Outputs:
        q_a: The smallest score that is larger than (n+1) * (1-alpha) of the other scores (= normal conformal qhat)
        q_b: The smallest score that is larger than (n+1) * (1-alpha) - 1 of the other scores
        gamma: value between 0 and 1 such that P(score <= gamma) * q_a + (1-gamma) * P(score <= g_b) = 1-alpha
    '''

    n = len(scores)
    
    if n == 0:
        return np.inf, 0, .1
    
    val_a = np.ceil((n+1)*(1-alpha)) / n
    if val_a > 1:
        q_a = default_qhat
    else:
        q_a = np.quantile(scores, val_a, method='inverted_cdf')
        
    val_b = (np.ceil((n+1)*(1-alpha)-1)) / n
    if val_b > 1:
        q_b = default_qhat
        
    else:
        q_b = np.quantile(scores, val_b, method='inverted_cdf') 
        
    if val_a > 1 and val_b > 1:
        gamma = 1 # Any value would work, since q_a and q_b are both equal to default_qhat
    else:
        overcov = np.ceil((n+1)*(1-alpha))/(n+1) - (1-alpha) # How much coverage using q_a will exceed 1 - alpha
        undercov = (1-alpha) - (np.ceil((n+1)*(1-alpha))-1)/(n+1)  #  How much coverage using q_b will undershoot 1 - alpha
        gamma = undercov / (undercov + overcov)

    return q_a, q_b, gamma

def construct_exact_coverage_standard_sets(q_a, q_b, gamma, scores_all):
    '''
    Construct randomized standard conformal sets
    
    Inputs:
        - q_a, q_b, gamma: as output by get_exact_coverage_conformal_params()
        - scores_all: num_instances x num_classes array
    '''
    # In non-randomized conformal, the qhat vector is fixed. But now, it will vary for each validation example
      
    scores_all = np.array(scores_all)
    set_preds = []
    num_samples = len(scores_all)
    for i in range(num_samples):
        if np.random.rand() < gamma: # Bernoulli random var
            q_hat = q_a
        else:
            q_hat = q_b
        set_preds.append(np.where(scores_all[i,:] <= q_hat)[0]) 

    return set_preds

def compute_exact_coverage_class_specific_params(scores_all, labels, num_classes, alpha, 
                                 default_qhat=np.inf, null_params=None):
    '''
    Compute the quantities necessary for performing classwise conformal prediction with exact coverage
    
    Inputs:
        - scores_all: (num_instances, num_classes) array where scores_all[i,j] = score of class j for instance i
        - labels: (num_instances,) array of true class labels
        - num_classes: Number of classes
        - alpha: number between 0 and 1 specifying desired coverage level
        - default_qhat: Quantile that will be used when there is insufficient data to compute a quantile
        - null_params: Dict of {'q_a': ..., 'q_b': ...., 'gamma', ...} to be assigned to
               class -1 (will be appended to end of param lists). Not needed if -1 does not appear in labels
    '''
    
    q_as = np.zeros((num_classes,))   
    q_bs = np.zeros((num_classes,)) 
    gammas = np.zeros((num_classes,)) 
    for k in range(num_classes):
        # Only select data for which k is true class
        
        idx = (labels == k)
        if len(scores_all.shape) == 2:
            scores = scores_all[idx, k]
        else:
            scores = scores_all[idx]
        
        q_a, q_b, gamma = get_exact_coverage_conformal_params(scores, alpha, default_qhat=default_qhat)
        q_as[k] = q_a
        q_bs[k] = q_b
        gammas[k] = gamma
        
    if -1 in labels:
        q_as = np.concatenate((q_as, [null_params['q_a']]))
        q_bs = np.concatenate((q_bs, [null_params['q_b']]))
        gamma = np.concatenate((gammas, [null_params['gamma']]))
        
    return q_as, q_bs, gammas
  
def construct_exact_coverage_classwise_sets(q_as, q_bs, gammas, scores_all):
    # In non-randomized conformal, the qhat vector is fixed. But now, it will vary for each validation example
      
    scores_all = np.array(scores_all)
    set_preds = []
    num_samples = len(scores_all)
    for i in range(num_samples):
        Bs = np.random.rand(len(gammas)) < gammas # Bernoulli random vars
        q_hats = np.where(Bs, q_as, q_bs)
        set_preds.append(np.where(scores_all[i,:] <= q_hats)[0]) 

    return set_preds

#========================================
#   Adaptive Prediction Sets (APS)
#========================================

def get_APS_scores(softmax_scores, labels, randomize=True, seed=0):
    '''
    Compute conformity score defined in Romano et al, 2020
    (Including randomization, unless randomize is set to False)
    
    Inputs:
        softmax_scores: n x num_classes
        labels: length-n array of class labels
    
    Output: 
        length-n array of APS scores
    '''
    n = len(labels)
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))[range(n), labels]
    scores = np.array(scores)
    
    if not randomize:
        return scores - softmax_scores[range(n), labels]
    else:
        np.random.seed(seed)
        U = np.random.rand(n) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores[range(n), labels]
        return randomized_scores
    
def get_APS_scores_all(softmax_scores, randomize=True, seed=0):
    '''
    Similar to get_APS_scores(), except the APS scores are computed for all 
    classes instead of just the true label
    
    Inputs:
        softmax_scores: n x num_classes
    
    Output: 
        n x num_classes array of APS scores
    '''
    n = softmax_scores.shape[0]
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))
    scores = np.array(scores)
    
    if not randomize:
        return scores - softmax_scores
    else:
        np.random.seed(seed)
        U = np.random.rand(*softmax_scores.shape) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores # [range(n), labels]
        return randomized_scores

#========================================
#   Regularized Adaptive Prediction Sets (RAPS)
#========================================

def get_RAPS_scores(softmax_scores, labels, lmbda=.01, kreg=5, randomize=True, seed=0):
    '''
    Essentially the same as get_APS_scores() except with regularization.
    See "Uncertainty Sets for Image Classifiers using Conformal Prediction" (Angelopoulos et al., 2021)
    
    Inputs:
        softmax_scores: n x num_classes
        labels: length-n array of class labels
        lmbda, kreg: regularization parameters
    Output: 
        length-n array of APS scores
    
    '''
    n = len(labels)
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))[range(n), labels]
    
    # Regularization
    y_rank = pi.argsort(1)[range(labels_calib.shape[0]), labels_calib] + 1 # Compute softmax rank of true labels y
    reg = torch.maximum(lmbda * (y_rank - kreg), torch.zeros(size=y_rank.shape))
    scores += reg
    
    scores = np.array(scores)
    
    if not randomize:
        return scores - softmax_scores[range(n), labels]
    else:
        np.random.seed(seed)
        U = np.random.rand(n) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores[range(n), labels]
        return randomized_scores
        
def get_RAPS_scores_all(softmax_scores, lmbda, kreg, randomize=True, seed=0):
    '''
    Similar to get_RAPS_scores(), except the RAPS scores are computed for all 
    classes instead of just the true label
    
    Inputs:
        softmax_scores: n x num_classes
    
    Output: 
        n x num_classes array of APS scores
    '''
    n = softmax_scores.shape[0]
    sorted, pi = torch.from_numpy(softmax_scores).sort(dim=1, descending=True) # pi is the indices in the original array
    scores = sorted.cumsum(dim=1).gather(1,pi.argsort(1))
    
    # Regularization (pretend each class is true label)
    y_rank = pi.argsort(1) + 1 # Compute softmax rank of true labels y
    reg = torch.maximum(lmbda * (y_rank - kreg), torch.zeros(size=scores.shape))
 
    scores += reg
        
    if not randomize:
        return scores - softmax_scores
    else:
        np.random.seed(seed)
        U = np.random.rand(*softmax_scores.shape) # Generate U's ~ Unif([0,1])
        randomized_scores = scores - U * softmax_scores # [range(n), labels]
        return randomized_scores

    
#========================================
#   Evaluation
#========================================

# Helper function for computing accuracy (marginal coverage) of confidence sets
def compute_coverage(true_labels, set_preds):
    true_labels = np.array(true_labels) # Convert to numpy to avoid weird pytorch tensor issues
    num_correct = 0
    for true_label, preds in zip(true_labels, set_preds):
        if true_label in preds:
            num_correct += 1
    set_pred_acc = num_correct / len(true_labels)
    
    return set_pred_acc

# Note: functionality is subsumed by compute_class_specific_metrics
# Helper function for computing class-conditional coverage of confidence sets
def compute_class_specific_coverage(true_labels, set_preds, return_class_cts=False):
    num_classes = max(true_labels) + 1
    class_specific_cov = np.zeros((num_classes,))
    class_cts = np.zeros((num_classes,))
    for k in range(num_classes):
        idx = np.where(true_labels == k)[0]
        selected_preds = [set_preds[i] for i in idx]
        num_correct = np.sum([1 if np.any(pred_set == k) else 0 for pred_set in selected_preds])
        class_specific_cov[k] = num_correct / len(selected_preds)
        class_cts[k] = np.sum(idx)

    if return_class_cts:
        return class_specific_cov, class_cts
        
    return class_specific_cov

def compute_class_specific_metrics(true_labels, set_preds):
    '''
    Computes class-specific coverage, class-specific average set size (average set size when class
    k is the true label), and class counts (# of times class k is the true label)
    '''
    num_classes = max(true_labels) + 1
    class_specific_cov = np.zeros((num_classes,))
    class_specific_avg_set_size = np.zeros((num_classes,))
    class_cts = np.zeros((num_classes,))
    for k in range(num_classes):
        idx = np.where(true_labels == k)[0]
        selected_preds = [set_preds[i] for i in idx]
        num_correct = np.sum([1 if np.any(pred_set == k) else 0 for pred_set in selected_preds])
        class_specific_cov[k] = num_correct / len(selected_preds)
        class_specific_avg_set_size[k] = compute_avg_set_size(selected_preds)
        class_cts[k] = np.sum(idx)


    return class_specific_cov, class_specific_avg_set_size, class_cts


def compute_class_false_positive_ct(true_labels, set_preds):
    '''
    Computes class false positive count (# of times class k is not the true label but 
    it is included in the prediction set)
    '''

    # v1: Slow
    # num_classes = max(true_labels) + 1
    # class_false_positive_ct = np.zeros((num_classes,))
    # for k in range(num_classes):
    #     not_k_pred_sets = [set_preds[i] for i in np.where(true_labels != k)[0]]
    #     class_false_positive_ct[k] = np.sum([k in set_pred for set_pred in not_k_pred_sets])

    # v2: Fast
   
    num_samples = len(true_labels)
    num_classes = max(true_labels) + 1
    
    # Initialize a sparse matrix in LIL format
    M = lil_matrix((num_samples, num_classes), dtype=int)
    
    # Populate the sparse matrix
    for i, pred_set in enumerate(set_preds):
        for cls in pred_set:
            M[i, cls] = 1
    
    # Convert to CSR format for efficient row slicing
    M = M.tocsr()
    
    # Initialize the false positive count array
    class_false_positive_ct = np.zeros(num_classes, dtype=int)
    
    # Compute false positives for each class
    for k in range(num_classes):
        # Find indices where the true label is not k
        not_k_indices = np.where(true_labels != k)[0]
        # Sum the entries in column k for these indices
        class_false_positive_ct[k] = M[not_k_indices, k].sum()
    

    return class_false_positive_ct

    
def compute_rarity_conditional_coverage(pred_sets, labels, show_plot=False, plot_params={}):
    '''
    Computes rarity conditional coverage. 

    Inputs:
        - pred_sets: list of lists
        - labels: array of same length as pred_sets
        - plot_params: dict that can contain 'plot_title', 'ylim'

    Output:
        rarity_cond_coverage: dict from num_examples to the coverage computed on all classes with that number of examples
        bin_ct: number of examples in each rarity bin
        avg_rarity_cond_miscov
        sd_of_cov
        
    '''

    # Count occurrences of each class
    class_counts = {k: np.sum(labels == k) for k in range(max(labels)+1)}

    rarity_cond_coverage = defaultdict(list) # dict with default value of an empty list
    

    # Compute per-example coverage
    for pred_set, label in zip(pred_sets, labels):
        num_examples = class_counts[label]  # Get frequency of the class
        coverage = int(label in pred_set)  # 1 if correctly covered, 0 otherwise
        rarity_cond_coverage[num_examples].append(coverage)

    bin_ct = {}
    for ct in class_counts.values():
        bin_ct[ct] = bin_ct.get(ct, 0) + ct

    ## Compute mean coverage for each rarity level
    rarity_cond_coverage = {num: np.mean(cov) for num, cov in rarity_cond_coverage.items()}
    rarity_cond_cov_arr = [v for v in rarity_cond_coverage.values()]

    # Aggregate the bin-conditional coverages into a single number
    # NOTE: not obvious what the best way to do this is. For now, I'll take a 
    # weighted average of the miscoverage rate where the weight for a bin is proportional to the number of examples in the bin

    # # v1: (Update: this is equivalent to marginal coverage)
    avg_rarity_cond_miscov = np.sum([(1 - rarity_cond_coverage[num_ex]) * bin_ct[num_ex] 
                                     for num_ex in list(rarity_cond_coverage.keys())]) / np.sum(list(bin_ct.values()))

    # # v2: unweighted average
    # avg_rarity_cond_miscov = 1 - np.mean(rarity_cond_cov_arr)

    ## Compute variance of coverage 
    sd_of_cov = np.std(rarity_cond_cov_arr)

    ## Compute mean of (1 - rarity-conditional miscoverage)
    mean_squared_miscov = np.mean((1 - np.array(rarity_cond_cov_arr)) ** 2)

    if show_plot:
        # Test on the double thresholded sets        
        plt.scatter(rarity_cond_coverage.keys(), rarity_cond_coverage.values(), s=np.array(list(bin_ct.values())) / 10, alpha=0.4)
        plt.xlabel('Num examples in class')
        plt.ylabel('Coverage for classes with given number of examples')
        plt.ylim(plot_params.get('ylim', (0,1)))
        plt.title(plot_params.get('plot_title', None))
        plt.show()

    return rarity_cond_coverage, bin_ct, avg_rarity_cond_miscov, sd_of_cov, mean_squared_miscov


# Helper function for computing average set size
def compute_avg_set_size(list_of_arrays):
    return np.mean([len(arr) for arr in list_of_arrays])

# Helper function for computing median set size
def compute_median_set_size(list_of_arrays):
    return np.median([len(arr) for arr in list_of_arrays])

def compute_all_metrics(val_labels, preds, alpha, cluster_assignments=None):
    class_cond_cov = compute_class_specific_coverage(val_labels, preds)
        
    # Average class coverage gap
    avg_class_cov_gap = np.mean(np.abs(class_cond_cov - (1-alpha)))

    # Average gap for classes that are over-covered
    overcov_idx = (class_cond_cov > (1-alpha))
    overcov_gap = np.mean(class_cond_cov[overcov_idx] - (1-alpha))

    # Average gap for classes that are under-covered
    undercov_idx = (class_cond_cov < (1-alpha))
    # undercov_gap = np.mean(np.abs(class_cond_cov[undercov_idx] - (1-alpha))) # PREVIOUS
    num_classes = np.max(val_labels)
    undercov_gap = np.sum(1-alpha - class_cond_cov[undercov_idx]) / num_classes # AVERAGED OVER ALL CLASSES
    
    # Fraction of classes that are at least 10% under-covered
    thresh = .1
    very_undercovered = np.mean(class_cond_cov < (1-alpha-thresh))

    # Fraction of classes with coverage below 50%
    cov_below50 = np.mean(class_cond_cov < 0.5)

    # Fraction of classes with coverage below 10%
    cov_below10 = np.mean(class_cond_cov < 0.1)
    
    # Max gap
    max_gap = np.max(np.abs(class_cond_cov - (1-alpha)))

    # Marginal coverage
    marginal_cov = compute_coverage(val_labels, preds)

    # Macro coverage (average within class, then average across classes)
    macro_cov = np.mean(class_cond_cov)

    class_cov_metrics = {'mean_class_cov_gap': avg_class_cov_gap, 
                         'undercov_gap': undercov_gap, 
                         'overcov_gap': overcov_gap, 
                         'max_gap': max_gap,
                         'very_undercovered': very_undercovered,
                         'cov_below50': cov_below50,
                         'cov_below10': cov_below10,
                         'macro_cov': macro_cov,
                         'marginal_cov': marginal_cov,
                         'raw_class_coverages': class_cond_cov,
                         'cluster_assignments': cluster_assignments # Also save class cluster assignments
                        }

    curr_set_sizes = [len(x) for x in preds]
    set_size_metrics = {'mean': np.mean(curr_set_sizes), '[.25, .5, .75, .9] quantiles': np.quantile(curr_set_sizes, [.25, .5, .75, .9])}
    
    # print('CLASS COVERAGE GAP:', avg_class_cov_gap)
    # print('AVERAGE SET SIZE:', np.mean(curr_set_sizes))
    
    return class_cov_metrics, set_size_metrics

#========================================
#   Misc.
#========================================

def convert_pred_sets_to_01(list_of_lists, num_classes):
    '''
    Convert list of prediction sets e.g. [[0,2], [5],...] to representation as matrix of 0 and 1s

    Input:
        - list_of_lists: length-n list of lists
        - num_classes: number of classes
    Output: M, an (n, num_classes) array of 0 and 1s, where M[i,j] = 1 if 
        class j appears in the ith prediction set
    '''
    n = len(list_of_lists)
    
    # M = lil_matrix((n, num_classes), dtype=int) # sparse representation
    M = np.zeros((n, num_classes), dtype=int)

    for i, pred_set in enumerate(list_of_lists):
        for cls in pred_set:
            M[i, cls] = 1

    return M