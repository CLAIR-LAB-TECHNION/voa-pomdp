from rocksample_experiments.utils import sample_problem_from_voa_row
import numpy as np
from scipy import stats


def test_heuristic_on_problem_instance(voa_df, env_instance_id, heuristic_function, heuristic_kwargs=None):
    """
    Test a heuristic function on a specific environment instance all helping actions,
    using efficient pandas operations
    """
    if heuristic_kwargs is None:
        heuristic_kwargs = {}

    # Filter for the specific environment instance
    env_voa_df = voa_df[voa_df['env_instance_id'] == env_instance_id].copy()

    # Apply heuristic function to each row using pandas apply
    def apply_heuristic(row):
        problem = sample_problem_from_voa_row(row, n=11)
        help_config = eval(row['help_actions'])
        return heuristic_function(problem, help_config, **heuristic_kwargs)

    # Calculate heuristic values
    env_voa_df['heuristic_value'] = env_voa_df.apply(apply_heuristic, axis=1)

    # Select relevant columns for the results
    results_df = env_voa_df[[
        'help_config_id',
        'heuristic_value',
        'empirical_voa',
        'ci_low_95',
        'ci_high_95',
        'ci_low_90',
        'ci_high_90',
        'ci_low_80',
        'ci_high_80'
    ]].copy()

    # Add environment instance ID
    results_df['env_instance_id'] = env_instance_id

    return results_df


def evaluate_top_k_accuracy(results_df, k=3):
    """
    Evaluate how many of top-k empirical VOA actions are identified by heuristic

    Args:
        results_df: DataFrame with columns ['empirical_voa', 'heuristic_value']
        k: Number of top actions to consider

    Returns:
        float: Proportion of top-k empirical actions that are in top-k heuristic predictions
    """
    top_k_empirical = set(results_df.nlargest(k, 'empirical_voa').index)
    top_k_heuristic = set(results_df.nlargest(k, 'heuristic_value').index)

    overlap = len(top_k_empirical.intersection(top_k_heuristic))
    accuracy = overlap / k

    return accuracy


def evaluate_rank_correlation(results_df):
    """
    Calculate Spearman's rank correlation between heuristic and empirical values

    Spearman's correlation assesses how well the heuristic preserves the ordering of actions
    compared to their empirical VOA values, regardless of the actual scale of predictions.

    Output range is [-1, 1]:
    - 1.0: Perfect ranking (heuristic orders actions exactly like empirical VOA)
    - 0.0: No correlation in rankings
    - -1.0: Completely reversed ranking

    Correlation strength guidelines:
    >0.7: Strong
    0.4-0.7: Moderate
    <0.4: Weak

    Returns:
        tuple: (correlation coefficient, p-value)
        p-value < 0.05 indicates statistical significance
    """
    correlation, p_value = stats.spearmanr(
        results_df['empirical_voa'],
        results_df['heuristic_value']
    )
    return correlation, p_value


def evaluate_sign_agreement(results_df, ci_type='95'):
    """
    Evaluate sign prediction accuracy considering confidence intervals.
    Only considers samples where empirical VOA is significantly different from zero.

    Args:
        results_df: DataFrame with empirical VOA and heuristic values
        ci_type: Confidence interval to use ('95', '90', or '80')

    Returns:
        dict containing:
        - precision: proportion of heuristic positive predictions that are actually positive
        - recall: proportion of true positives that are correctly identified
        - accuracy: proportion of correct predictions among all significant samples
        - n_significant: number of samples with significant VOA
        - n_total: total number of samples
    """
    if ci_type not in ['95', '90', '80']:
        raise ValueError("ci_type must be '95', '90', or '80'")

    # Identify significant VOA using specified CI
    significant_positive = (results_df[f'ci_low_{ci_type}'] > 0)
    significant_negative = (results_df[f'ci_high_{ci_type}'] < 0)

    # Only consider samples with significant VOA (either positive or negative)
    significant_mask = significant_positive | significant_negative
    significant_df = results_df[significant_mask]

    if len(significant_df) == 0:
        return {
            'precision': None,
            'recall': None,
            'accuracy': None,
            'n_significant': 0,
            'n_total': len(results_df)
        }

    # Actual and predicted labels for significant samples
    true_labels = significant_df[f'ci_low_{ci_type}'] > 0
    predicted_labels = significant_df['heuristic_value'] > 0

    # Calculate metrics
    true_positives = np.sum(true_labels & predicted_labels)
    false_positives = np.sum(~true_labels & predicted_labels)
    false_negatives = np.sum(true_labels & ~predicted_labels)
    true_negatives = np.sum(~true_labels & ~predicted_labels)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(significant_df)

    return {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'n_significant': len(significant_df),
        'n_total': len(results_df)
    }


def evaluate_ci_weighted_correlation(results_df):
    """
    Calculate correlation weighted by confidence interval width
    """
    ci_widths = results_df['ci_high_95'] - results_df['ci_low_95']
    weights = 1 / ci_widths
    weights = weights / weights.sum()  # normalize weights

    weighted_corr = np.corrcoef(
        results_df['empirical_voa'] * weights,
        results_df['heuristic_value'] * weights
    )[0, 1]

    return weighted_corr


def heuristic_metrics(results_df):
    """
    Comprehensive evaluation of heuristic performance

    Args:
        results_df: DataFrame with results from test_heuristic_on_problem_instance

    Returns:
        dict with all evaluation metrics
    """
    evaluation = {
        'top_1_accuracy': evaluate_top_k_accuracy(results_df, k=1),
        'top_35_accuracy': evaluate_top_k_accuracy(results_df, k=5),
        'rank_correlation': evaluate_rank_correlation(results_df)[0],
        'rank_correlation_pvalue': evaluate_rank_correlation(results_df)[1],
        'sign_agreement': evaluate_sign_agreement(results_df),
        'ci_weighted_correlation': evaluate_ci_weighted_correlation(results_df)
    }

    return evaluation