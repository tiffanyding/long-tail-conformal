import numpy as np
from utils.conformal_utils import compute_rc3p_params, create_rc3p_prediction_sets

def test_rc3p_debug():
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
    import pdb; pdb.set_trace()

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

if __name__ == "__main__":
    test_rc3p_debug()