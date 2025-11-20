import numpy as np
from utils.conformal_utils import compute_rc3p_params, create_rc3p_prediction_sets

def test_rc3p_small():
    """
    Test RC3P with a small synthetic dataset and add a debugging point to inspect ranks computation.
    """
    # Calibration data
    cal_softmax_scores = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8],
        [0.6, 0.3, 0.1]
    ])
    cal_scores_all = 1 - cal_softmax_scores  # Example: softmax scores as base scores
    cal_labels = np.array([0, 1, 2, 0])

    # Test data
    test_softmax_scores = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.6, 0.1],
        [0.1, 0.2, 0.7]
    ])
    test_scores_all = 1 - test_softmax_scores
    test_labels = np.array([0, 1, 2])

    alpha = 0.1

    # Compute RC3P parameters
    q_hats_rc3p, k_hats, alpha_hats = compute_rc3p_params(
        cal_softmax_scores=cal_softmax_scores,
        cal_scores_all=cal_scores_all,
        cal_labels=cal_labels,
        alpha=alpha
    )


    # Debugging point: Inspect ranks computation

    # Create RC3P prediction sets
    rc3p_preds = create_rc3p_prediction_sets(
        softmax_scores=test_softmax_scores,
        scores_all=test_scores_all,
        q_hats_rc3p=q_hats_rc3p,
        k_hats=k_hats
    )

    # Assertions
    assert len(rc3p_preds) == len(test_labels), "Number of predictions should match the number of test examples."
    assert all(isinstance(pred, np.ndarray) for pred in rc3p_preds), "Each prediction set should be a numpy array."

def test_rc3p_big():

    # Generate softmax array for 100 classes
    num_classes = 100
    num_calibration = 5000
    num_test = 1000
    np.random.seed(42)
    cal_softmax_scores = np.random.dirichlet(np.ones(num_classes), size=num_calibration)
    test_softmax_scores = np.random.dirichlet(np.ones(num_classes), size=num_test)
    cal_scores_all = 1 - cal_softmax_scores
    test_scores_all = 1 - test_softmax_scores
    cal_labels = np.random.randint(0, num_classes, size=num_calibration)
    test_labels = np.random.randint(0, num_classes, size=num_test)  
    alpha = 0.1

    # Compute RC3P parameters
    q_hats_rc3p, k_hats, alpha_hats = compute_rc3p_params(
        cal_softmax_scores=cal_softmax_scores,
        cal_scores_all=cal_scores_all,
        cal_labels=cal_labels,
        alpha=alpha
    )
    # Create RC3P prediction sets
    rc3p_preds = create_rc3p_prediction_sets(
        softmax_scores=test_softmax_scores,
        scores_all=test_scores_all,
        q_hats_rc3p=q_hats_rc3p,
        k_hats=k_hats
    )

    # Compute class-conditional and marginal coverage
    from utils.conformal_utils import compute_all_metrics
    coverage_metrics, set_size_metrics = compute_all_metrics(test_labels, rc3p_preds,
                                                            alpha)
    print("RC3P Coverage Metrics:", coverage_metrics)
    print("RC3P Set Size Metrics:", set_size_metrics)


if __name__ == "__main__":
    # test_rc3p_small()

    test_rc3p_big()