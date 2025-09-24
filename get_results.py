import argparse
import os
import sys
import time

from utils.conformal_utils import *
from utils.experiment_utils import get_inputs_folder, get_outputs_folder
# from baselines_utils import *
# from train_models.train import get_datasets

'''
Usage:

    python get_results.py plantnet

OR
    python get_results.py plantnet-trunc

OR
    python get_results.py inaturalist

OR

    python get_results.py inaturalist-trunc

'''

def prep_for_save(res, labels, alpha, train_labels_path):
    pred_sets = res['pred_sets']
    coverage_metrics, set_size_metrics = compute_all_metrics(labels, pred_sets, alpha)

    ### Add additional metrics

    # Array of coverage indicators
    coverage_metrics['is_covered'] = np.array([label in pred_set for label, pred_set in zip(labels, pred_sets)])

    # Array of set sizes
    coverage_metrics['raw_set_sizes'] = np.array([len(pred_set) for pred_set in pred_sets])

    num_classes = np.max(labels) + 1
    val_class_distr = np.array([np.sum(labels == k) for k in range(num_classes)]) / len(labels)
    train_labels = np.load(train_labels_path)
    train_class_distr = np.array([np.sum(train_labels == k) for k in range(num_classes)]) / len(train_labels) 

    # Marginal coverage, wrt to val distribution (necessary for truncated datasets where test distr != true distr)
    coverage_metrics['val_marginal_cov'] = np.sum([val_class_distr[k] * coverage_metrics['raw_class_coverages'][k] for k in range(num_classes)])

    # Marginal coverage, wrt to train distribution (necessary for truncated datasets where test distr != true distr)
    coverage_metrics['train_marginal_cov'] = np.sum([train_class_distr[k] * coverage_metrics['raw_class_coverages'][k] for k in range(num_classes)])

    # Modify in place: add metrics, delete prediction set to save space
    res['coverage_metrics'] = coverage_metrics
    res['set_size_metrics'] = set_size_metrics
    res.pop('pred_sets', None)

def save(res, path):  
    with open(path, 'wb') as f:
        pickle.dump(res, f)
    print(f'Saved res', path)


def get_results(dataset, alphas, methods, score='softmax', results_folder='results', 
                model_type='best', loss='cross_entropy', override_saved=False):
    '''
    model_type: 'best' or 'last_epoch'
    override_saved: whether to override saved results (if False, we skip recomputing)
    '''
    print(f'Results will be computed for: \n{dataset=}\n{alphas=}\n{methods=}')
    os.makedirs(results_folder, exist_ok=True)

    # ------- Get data --------
    # folder = '/home-warm/plantnet/conformal_cache/train_models'
    folder = get_inputs_folder()
    if loss != 'cross_entropy':
        folder = os.path.join(folder, loss)
    model_type = model_type.replace('_', '-')
    print(f'Loading {model_type} model softmax scores and labels from', folder)
    split = 'cal'
        
    val_softmax = np.load(f'{folder}/{model_type}-{dataset}-model_{split}_softmax.npy')
    val_labels = np.load(f'{folder}/{model_type}-{dataset}-model_{split}_labels.npy')
    test_softmax = np.load(f'{folder}/{model_type}-{dataset}-model_test_softmax.npy')
    test_labels = np.load(f'{folder}/{model_type}-{dataset}-model_test_labels.npy')
    print('Loaded pre-computed softmax scores')
    
    num_classes = val_softmax.shape[1]
    print('Num classes:', num_classes)

    train_labels_path = f'{folder}/{dataset}_train_labels.npy'

    val_scores = get_conformal_scores(val_softmax, score, train_labels_path=train_labels_path)
    test_scores = get_conformal_scores(test_softmax, score, train_labels_path=train_labels_path) 

    for alpha in alphas:
        print(f'\n==== COMPUTING RESULTS FOR {dataset}, alpha={alpha} ====')
        results_prefix = f'{results_folder}/{dataset}_{score}_alpha={alpha}'

    
        # ------- Run baseline methods --------
        ## Standard CP 
        if 'standard' in methods: 
            save_path = f'{results_prefix}_standard.pkl'
            if not os.path.exists(save_path) or override_saved:
                standard_qhat, pred_sets, _, _ = standard_conformal(val_scores, val_labels, 
                                                           test_scores, test_labels, alpha)
                res = {'pred_sets': pred_sets, 'qhat': standard_qhat}
                prep_for_save(res, test_labels, alpha, train_labels_path)
                save(res, save_path)
        
        ## Classwise CP
        if 'classwise' in methods:
            save_path = f'{results_prefix}_classwise.pkl'
            if not os.path.exists(save_path) or override_saved:
                classwise_qhats, pred_sets, _, _ = classwise_conformal(val_scores, val_labels, 
                                                           test_scores, test_labels, alpha,
                                                           num_classes, default_qhat=np.inf)
                
                res = {'pred_sets': pred_sets, 'qhats': classwise_qhats}
                prep_for_save(res, test_labels, alpha, train_labels_path)
                save(res, save_path)
    
        
        # ## Classwise CP with randomization to achieve exact coverage
        if 'classwise-exact' in methods:
            save_path = f'{results_prefix}_classwise-exact.pkl'
            if not os.path.exists(save_path) or override_saved:
                qhats, pred_sets, _, _ = classwise_conformal(val_scores, val_labels, 
                                                           test_scores, test_labels, alpha,
                                                           num_classes, default_qhat=np.inf, exact_coverage=True)
                res = {'pred_sets': pred_sets, 'qhats': qhats}
                prep_for_save(res, test_labels, alpha, train_labels_path)
                save(res, save_path)
        
        ## Clustered CP
        if 'clustered' in methods:
            save_path = f'{results_prefix}_clustered.pkl'
            if not os.path.exists(save_path) or override_saved:
                qhats, pred_sets, _, _ = clustered_conformal(val_scores, val_labels, alpha,
                                                            test_scores, test_labels)
                res = {'pred_sets': pred_sets, 'qhats': qhats}
                prep_for_save(res, test_labels, alpha, train_labels_path)
                save(res, save_path)

        # ------- Run Macro Softmax --------
        # Note: this is the same regardless of what we passed in for "score", because it overrides the score
        # with the prevalence-adjusted softmax

        # Compute the new score
        if 'prevalence-adjusted' in methods:
            save_path = f'{results_prefix}_prevalence-adjusted.pkl'
            if not os.path.exists(save_path) or override_saved:    
                train_labels = np.load(train_labels_path)
                train_class_distr = np.array([np.sum(train_labels == k) for k in range(num_classes)]) / len(train_labels) 
                val_scores_prevalence = 1 - (val_softmax / train_class_distr)
                test_scores_prevalence = 1 - (test_softmax / train_class_distr)
        
                qhat, pred_sets, _, _ = standard_conformal(val_scores_prevalence, val_labels, 
                                                           test_scores_prevalence, test_labels, alpha)
                res = {'pred_sets': pred_sets, 'qhat': qhat}
                prep_for_save(res, test_labels, alpha, train_labels_path)
                save(res, save_path)
    
        # ------- Run Fuzzy CP methods --------
        ## Random projection ("Baseline") 
        if 'fuzzy-random' in methods:
            random_bandwidths = [1e-30, 1e-15, 1e-10, 1e-5, 0.0001, 0.001, 0.01, .1 , 10, 1000]
            for bandwidth in random_bandwidths:
                save_path = f'{results_prefix}_fuzzy-random-{bandwidth}.pkl'
                if not os.path.exists(save_path) or override_saved: 
                    qhats, pred_sets, proj_arr = fuzzy_classwise_CP(val_scores, val_labels, alpha, 
                                                          val_scores_all=test_scores, 
                                                          projection='random', mode='weight', 
                                                          params={'bandwidth': bandwidth}, show_diagnostics=False)
                    res = {'pred_sets': pred_sets, 'qhats': qhats, 'proj_arr': proj_arr}
                    prep_for_save(res, test_labels, alpha, train_labels_path)
                    save(res, save_path)
        
        ## Rarity projection 
        if 'fuzzy-rarity' in methods:
            rarity_bandwidths = [1e-30, 1e-15, 1e-10, 1e-5, 0.0001, 0.001, 0.01, .1 , 10, 1000]
            for bandwidth in rarity_bandwidths:
                save_path = f'{results_prefix}_fuzzy-rarity-{bandwidth}.pkl'
                if not os.path.exists(save_path) or override_saved: 
                    qhats, pred_sets, proj_arr = fuzzy_classwise_CP(val_scores, val_labels, alpha, 
                                                          val_scores_all=test_scores, 
                                                          projection='rarity', mode='weight', 
                                                          params={'bandwidth': bandwidth, 
                                                                  'use_train': True, 
                                                                  'train_labels_path': train_labels_path, 
                                                                  'dataset': dataset}, show_diagnostics=False)
                    res = {'pred_sets': pred_sets, 'qhats': qhats, 'proj_arr': proj_arr}
                    prep_for_save(res, test_labels, alpha, train_labels_path)
                    save(res, save_path)
        
        ## Quantile projection 
        if 'fuzzy-quantile' in methods:
            QP_bandwidths = [1e-30, 1e-15, 1e-10, 1e-5, 0.0001, 0.001, 0.01, .1 , 10, 1000]
            for bandwidth in QP_bandwidths:
                save_path = f'{results_prefix}_fuzzy-quantile-{bandwidth}.pkl'
                if not os.path.exists(save_path) or override_saved: 
                    qhats, pred_sets, proj_arr = fuzzy_classwise_CP(val_scores, val_labels, alpha, 
                                                          val_scores_all=test_scores, 
                                                          projection='quantile', mode='weight', 
                                                          params={'bandwidth': bandwidth}, show_diagnostics=False)
                    res = {'pred_sets': pred_sets, 'qhats': qhats, 'proj_arr': proj_arr}
                    prep_for_save(res, test_labels, alpha, train_labels_path)
                    save(res, save_path)
                    
        # ----- Reconformalized variants ------
        reconformalize = 'alpha' # 'additive' or 'multiplicative' or 'alpha'
        
        # Create holdout dataset for reconformalization
        num_holdout = 5000 
        print(f'Holding out {num_holdout} of {len(val_labels)} examples for reconformalization step')
        shuffle_idx = np.random.permutation(np.arange(len(val_labels)))
        holdout_idx = shuffle_idx[:num_holdout]
        cal_idx = shuffle_idx[num_holdout:]
        
        cal_scores_all, cal_labels = val_scores[cal_idx], val_labels[cal_idx]
        holdout_scores_all, holdout_labels = val_scores[holdout_idx], val_labels[holdout_idx]
        
        
        # RandProj 
        if 'fuzzy-RErandom' in methods:
            random_bandwidths = [1e-30, 1e-15, 1e-10, 1e-5, 0.0001, 0.001, 0.01, .1 , 10, 1000]
            for bandwidth in random_bandwidths:
                save_path = f'{results_prefix}_fuzzy-RErandom-{bandwidth}.pkl'
                if not os.path.exists(save_path) or override_saved: 
                    qhats, pred_sets, proj_arr = fuzzy_classwise_CP(cal_scores_all, cal_labels, alpha, 
                                                          val_scores_all=test_scores, 
                                                          projection='random', mode='weight', 
                                                          reconformalize=reconformalize,
                                                          reconformalize_data=(holdout_scores_all, holdout_labels),
                                                          params={'bandwidth': bandwidth}, show_diagnostics=False)
                    res = {'pred_sets': pred_sets, 'qhats': qhats, ' proj_arr':  proj_arr}
                    prep_for_save(res, test_labels, alpha, train_labels_path)
                    save(res, save_path)
                    
        # Rarity
        if 'fuzzy-RErarity' in methods:
            rarity_bandwidths = [1e-30, 1e-15, 1e-10, 1e-5, 0.0001, 0.001, 0.01, .1 , 10, 1000]
            for bandwidth in rarity_bandwidths:
                save_path = f'{results_prefix}_fuzzy-RErarity-{bandwidth}.pkl'
                if not os.path.exists(save_path) or override_saved: 
                    qhats, pred_sets, proj_arr = fuzzy_classwise_CP(cal_scores_all, cal_labels, alpha, 
                                                          val_scores_all=test_scores, 
                                                          projection='rarity', mode='weight', 
                                                          reconformalize=reconformalize,
                                                          reconformalize_data=(holdout_scores_all, holdout_labels),
                                                          params={'bandwidth': bandwidth, 
                                                                  'use_train': True, 
                                                                  'train_labels_path': train_labels_path, 
                                                                  'dataset': dataset}, show_diagnostics=False)
                    res = {'pred_sets': pred_sets, 'qhats': qhats, ' proj_arr':  proj_arr}
                    prep_for_save(res, test_labels, alpha, train_labels_path)
                    save(res, save_path)

        # Rarity with additive reconformalization
        if 'fuzzy-READDrarity' in methods:
            rarity_bandwidths = [1e-30, 1e-15, 1e-10, 1e-5, 0.0001, 0.001, 0.01, .1 , 10, 1000]
            for bandwidth in rarity_bandwidths:
                save_path = f'{results_prefix}_fuzzy-READDrarity-{bandwidth}.pkl'
                if not os.path.exists(save_path) or override_saved:
                    qhats, pred_sets, proj_arr = fuzzy_classwise_CP(cal_scores_all, cal_labels, alpha, 
                                                          val_scores_all=test_scores, 
                                                          projection='rarity', mode='weight', 
                                                          reconformalize='additive',
                                                          reconformalize_data=(holdout_scores_all, holdout_labels),
                                                          params={'bandwidth': bandwidth, 
                                                                  'use_train': True, 
                                                                  'train_labels_path': train_labels_path, 
                                                                  'dataset': dataset}, show_diagnostics=False)
                    res = {'pred_sets': pred_sets, 'qhats': qhats, ' proj_arr':  proj_arr}
                    prep_for_save(res, test_labels, alpha, train_labels_path)
                    save(res, save_path)
        
        # QuantileProj
        if 'fuzzy-REquantile' in methods:
            QP_bandwidths = [1e-30, 1e-15, 1e-10, 1e-5, 0.0001, 0.001, 0.01, .1 , 10, 1000]
            for bandwidth in QP_bandwidths:
                save_path = f'{results_prefix}_fuzzy-REquantile-{bandwidth}.pkl'
                if not os.path.exists(save_path) or override_saved: 
                    qhats, pred_sets, proj_arr = fuzzy_classwise_CP(cal_scores_all, cal_labels, alpha, 
                                                          val_scores_all=test_scores, 
                                                          projection='quantile', mode='weight', 
                                                          reconformalize=reconformalize,
                                                          reconformalize_data=(holdout_scores_all, holdout_labels),
                                                          params={'bandwidth': bandwidth}, show_diagnostics=False)
                    res = {'pred_sets': pred_sets, 'qhats': qhats, ' proj_arr':  proj_arr}
                    prep_for_save(res, test_labels, alpha, train_labels_path)
                    save(res, save_path)
                    
        # # ------- Run Convex Combination Methods --------
        if ('cvx' in methods) or ('monotonic-cvx' in methods):
            # weights = 1 - np.array([0, .001, .01, .025, .05, .1, .15, .2, .4, .6, .8, 1])
            weights = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99 , 0.999, 1]
            with open(f'{results_prefix}_standard.pkl', 'rb') as f:
                standard_res = pickle.load(f)
                standard_qhat = standard_res['qhat']
            with open(f'{results_prefix}_classwise.pkl', 'rb') as f:
                classwise_res = pickle.load(f)
                cw_qhats = classwise_res['qhats']    
            cw_qhats[np.isinf(cw_qhats)] = 1
        
        # Original convex combination
        if 'cvx' in methods:
            for w_cw in weights:
                save_path = f'{results_prefix}_cvx-cw_weight={w_cw}.pkl'
                if not os.path.exists(save_path) or override_saved: 
                    cvx_classwise_qhats = w_cw * cw_qhats + (1 - w_cw) * standard_qhat
                    cvx_classwise_pred_sets = create_classwise_prediction_sets(test_scores, cvx_classwise_qhats)
                    res = {'pred_sets': cvx_classwise_pred_sets, 'qhats': cvx_classwise_qhats}
                    prep_for_save(res, test_labels, alpha, train_labels_path)
                    save(res, save_path)

        # # "Monotonic" convex combination (only downwards)
        if 'monotonic-cvx' in methods:
            keep_as_cw = cw_qhats <= standard_qhat
            print(f"# of classes with classwise qhat larger than standard qhat: {np.sum(keep_as_cw)}")
            for w_cw in weights:
                save_path = f'{results_prefix}_monotonic-cvx-cw_weight={w_cw}.pkl'
                if not os.path.exists(save_path) or override_saved: 
                    cvx_classwise_qhats = w_cw * cw_qhats + (1 - w_cw) * standard_qhat
                    cvx_classwise_qhats[keep_as_cw] = cw_qhats[keep_as_cw]
                    cvx_classwise_pred_sets = create_classwise_prediction_sets(test_scores, cvx_classwise_qhats)
                    res = {'pred_sets': cvx_classwise_pred_sets, 'qhats': cvx_classwise_qhats}
                    prep_for_save(res, test_labels, alpha, train_labels_path)
                    save(res, save_path)

if __name__ == "__main__":
    st = time.time()
    
    parser = argparse.ArgumentParser(description='Get prediction sets')
    parser.add_argument('dataset', type=str, 
                        choices=['plantnet', 'inaturalist', 'plantnet-trunc', 'inaturalist-trunc'],
                        help='Name of the dataset')
    parser.add_argument('--score', type=str, choices=['softmax', 'APS', 'RAPS', 'PAS'], default='softmax',
                        help='Name of conformal score function')
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0.2, 0.1, 0.05, 0.01],
                        help='List of miscoverage levels (e.g. 0.2 0.1 0.05 0.01)')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=[
                          'standard', 'classwise', 'classwise-exact', 'clustered',
                          'fuzzy-rarity', 'fuzzy-RErarity', 'cvx', 'prevalence-adjusted'
                        ],
                        help='Which methods to run')
    parser.add_argument('--model_type', type=str, choices=['best', 'last_epoch', 'proper_cal'], default='best',
                        help='Whether to use weights from model with best val accuracy (where val is then reused as cal)' +
                               'OR the weights from the last epoc' +
                               'OR the best val-acc weights (where val is separate from cal)' )
    parser.add_argument('--loss', type=str, default='cross_entropy',
                    help='Loss function: Options are "cross_entropy" or "focal" (designed for imbalanced data)')
    parser.add_argument('--override_saved', action='store_true',
                help='If set, overrides existing saved results instead of skipping them, which is the default')

    

    args = parser.parse_args()

    score = args.score
    alphas  = args.alphas
    methods = args.methods
    model_type = args.model_type
    loss = args.loss
    override_saved = args.override_saved


    # alphas = [0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01]
    # alphas = [0.2, 0.1, 0.05, 0.01]
    # alphas = [.045, .04, .035, .03, .025, .02, .015]
    # methods = ['standard', 'classwise', 'classwise-exact', 'clustered', 'prevalence-adjusted', 
    #            'fuzzy-rarity', 'fuzzy-RErarity', 'fuzzy-READDrarity', 'cvx', 'monotonic-cvx'] # ATTN
    
    # # methods = ['fuzzy-rarity', 'fuzzy-RErarity']
    # score = 'softmax'

    # methods = ['standard', 'classwise', 'classwise-exact', 'clustered',  
    #            'fuzzy-rarity', 'fuzzy-RErarity', 'cvx'] 
    # score = 'PAS'

    results_folder = get_outputs_folder()
    if loss != 'cross_entropy':
        results_folder = os.path.join(results_folder, loss)
    
    if (model_type == 'last_epoch'):
        results_folder = os.path.join(results_folder, model_type)
    
    print('Results will be saved to', results_folder)
    get_results(args.dataset, alphas, methods, score, results_folder=results_folder, 
                model_type=model_type, loss=loss, override_saved=override_saved)

    print(f'Time taken: {(time.time() - st) / 60:.2f} minutes')
    
    