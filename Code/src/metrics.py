import numpy as np
from math import sqrt

#Performance evaluation

def pearson_correlation(y_preds, y_trues):
    """
    Calculates the Pearson correlation coefficient between two lists of data
    (predicted and true values).

    Args:
        y_preds (list): The list of numerical predicted values.
        y_trues (list): The list of numerical true values.

    Returns:
        float: The Pearson correlation coefficient.
               Returns float('nan') if:
               - Input lists have different lengths.
               - Length of lists is less than 2 (correlation is undefined).
               - One or both lists have zero variance (all elements are identical).
    """
    n_preds = len(y_preds)
    n_trues = len(y_trues)

    if n_preds != n_trues:
        print("Error: Input lists (y_preds and y_trues) must have the same length.")
        return float('nan')

    n = n_preds # Both lengths are equal here

    if n < 2:
        print("Error: Pearson correlation coefficient requires at least 2 data points.")
        return float('nan')

    # Calculate means
    # Using np.mean is robust even if len(y_preds) is 0, but we already handled n < 2
    mean_y_preds = np.mean(y_preds)
    mean_y_trues = np.mean(y_trues)

    # Calculate numerator: sum((y_preds_i - mean_y_preds) * (y_trues_i - mean_y_trues))
    numerator = 0
    for i in range(n):
        numerator += (y_preds[i] - mean_y_preds) * (y_trues[i] - mean_y_trues)

    # Calculate denominator components: sum((y_preds_i - mean_y_preds)^2) and sum((y_trues_i - mean_y_trues)^2)
    sum_sq_dev_y_preds = 0
    sum_sq_dev_y_trues = 0
    for i in range(n):
        sum_sq_dev_y_preds += (y_preds[i] - mean_y_preds)**2
    for i in range(n): # Corrected loop to be independent
        sum_sq_dev_y_trues += (y_trues[i] - mean_y_trues)**2

    denominator = sqrt(sum_sq_dev_y_preds * sum_sq_dev_y_trues)

    if denominator == 0:
        print("Warning: Cannot calculate Pearson correlation. One or both datasets have zero variance (all values are identical).")
        return float('nan')

    pearson_r = numerator / denominator

    return float(pearson_r)

def _get_ranks(data):
    """
    Assigns ranks to the elements in a list, handling ties by assigning the average rank.
    """
    if not data:
        return []

    # Create a list of (value, original_index) tuples
    indexed_data = sorted([(value, i) for i, value in enumerate(data)])
    
    ranks = [0] * len(data)
    i = 0
    while i < len(indexed_data):
        j = i
        # Find all tied values
        while j < len(indexed_data) and indexed_data[j][0] == indexed_data[i][0]:
            j += 1
        
        # Calculate the average rank for the tied values
        # Ranks are 1-based, so for 0-indexed original positions, (i+1 + j)/2
        # However, for the average rank, we take the average of 0-indexed positions
        # which effectively gives the average of (i, i+1, ..., j-1)
        avg_rank = (i + j - 1) / 2 + 1 # Convert to 1-based average rank

        # Assign the average rank to all tied elements
        for k in range(i, j):
            original_index = indexed_data[k][1]
            ranks[original_index] = avg_rank
        i = j
    return ranks

def spearman_correlation(y_preds, y_trues):
    """
    Calculates the Spearman's Rank Correlation Coefficient between two lists of data
    (predicted and true values) by leveraging the pearson_correlation function.

    Args:
        y_preds (list): The list of numerical predicted values.
        y_trues (list): The list of numerical true values.

    Returns:
        float: The Spearman correlation coefficient.
               Returns float('nan') if:
               - Input lists have different lengths.
               - Length of lists is less than 2 (correlation is undefined).
               - One or both lists (after ranking) have zero variance (all elements are identical).
                 This specifically covers cases where all y_preds or y_trues are identical.
    """
    n_preds = len(y_preds)
    n_trues = len(y_trues)

    if n_preds != n_trues:
        print("Error: Input lists (y_preds and y_trues) must have the same length for Spearman correlation.")
        return float('nan')

    n = n_preds # Both lengths are equal here

    if n < 2:
        print("Error: Spearman correlation coefficient requires at least 2 data points.")
        return float('nan')

    # Check if all predicted values are the same *before* ranking,
    # as ranking them would result in all ranks being the same (e.g., all 1.0s or avg rank)
    # which would then correctly lead to NaN from pearson_correlation, but this makes it explicit.
    if len(set(y_preds)) == 1:
        print("Warning: Cannot calculate Spearman correlation. All predicted values are identical.")
        return float('nan')

    # Similar check for true values, though usually y_trues will have variance
    if len(set(y_trues)) == 1:
        print("Warning: Cannot calculate Spearman correlation. All true values are identical.")
        return float('nan')

    # 1. Get ranks for both lists
    ranks_y_preds = _get_ranks(y_preds)
    ranks_y_trues = _get_ranks(y_trues)
    
    # 2. Calculate Pearson correlation on the ranks
    # The pearson_correlation function already handles cases where the ranks might have zero variance
    # (e.g., if all y_preds were identical, ranks_y_preds would all be the same, resulting in NaN)
    spearman_r = pearson_correlation(ranks_y_preds, ranks_y_trues)

    return spearman_r
    
def r2_score(y_preds, y_trues):
    y_trues = [float(y) for y in y_trues]
    y_preds = [float(y) for y in y_preds]
    true_mean = np.mean(y_trues)
    rss = sum((yt - yp) ** 2 for yp, yt in zip(y_preds, y_trues))
    tss = sum((yt - true_mean) ** 2 for yt in y_trues)
    EPSILON = 1e-9

    return 1 - (rss / (tss + EPSILON)) if tss == 0 else 1 - (rss / tss)

def rmse(y_preds, y_trues):
    y_preds = [float(y) for y in y_preds]
    y_trues = [float(y) for y in y_trues]
    n = len(y_preds)
    EPSILON = 1e-9
    mse = sum((yt - yp) ** 2 for yp, yt in zip(y_preds, y_trues)) / n
    return sqrt(mse + EPSILON) if mse == 0 else sqrt(mse)

def mse(y_preds, y_trues):
    y_preds = [float(y) for y in y_preds]
    y_trues = [float(y) for y in y_trues]
    n = len(y_preds)
    return sum((yt - yp) ** 2 for yp, yt in zip(y_preds, y_trues)) / n
    

def ndcg_score(y_trues, y_preds):

    y_trues = np.asarray(y_trues)
    y_preds = np.asarray(y_preds)

    if len(y_trues) != len(y_preds):
        raise ValueError("y_trues and y_preds must have the same length.")

    if not y_trues.size:
        return 0.0


    predicted_rank_indices = np.argsort(y_preds)[::-1]  # Get indices that would sort y_preds in descending order
    ranked_true_fitnesses = y_trues[predicted_rank_indices]

    # Calculating the DCG
    dcg = 0.0
    for i, fitness in enumerate(ranked_true_fitnesses):
        dcg += fitness / np.log2(i + 1 + 1)
        
    perfect_ranked_true_fitnesses = np.atleast_1d(np.sort(y_trues)[::-1])

    idcg = 0.0
    for i, fitness in enumerate(perfect_ranked_true_fitnesses):
        idcg += fitness / np.log2(i + 2) # Same denominator logic as above

    # 3. Calculate NDCG
    if idcg == 0.0:
        return 0.0  # Avoid division by zero, if there are no relevant items or all true fitnesses are 0
    else:
        ndcg = dcg / idcg
        return ndcg

# custom training metrics for lightgbm/xgboost
def spearman_lightgbm(y_trues, y_preds):
    return 'rho', spearman_correlation(y_trues, y_preds), True

def ndcg_lightgbm(y_trues, y_preds):
    return 'ndcg', ndcg_score(y_trues, y_preds), True

def spearman_xgboost(y_trues, y_preds):
    return 'rho', spearman_correlation(y_trues, y_preds)

def ndcg_xgboost(y_trues, y_preds):
    return 'ndcg', ndcg_score(y_trues, y_preds)


