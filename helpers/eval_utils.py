import pandas as pd
import numpy as np


def twohee_data_col_to_df(twohee_data_collection):
    res_list = twohee_data_collection.to_list()
    res_obj_list = []
    for r in res_list:
        res_obj = vars(r)
        res_obj_list.append(res_obj)
    res_df = pd.DataFrame(res_obj_list)
    # Add ground truth column
    res_df['ground_truth'] = res_df['rel_video_id'].apply(
        lambda x: int(x[len('video'):]))

    # TODO add conditional for frame_id here
    return res_df.copy()


def average_precision(ground_truth, predictions):
    """
    Calculate the Average Precision (AP) for a single query.

    Args:
        ground_truth (int): The ground truth video ID.
        predictions (list): List of predicted video IDs.

    Returns:
        float: The Average Precision (AP) score for the query.
    """
    hits = 0
    sum_precision = 0
    for i, pred in enumerate(predictions):
        if pred == ground_truth:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / hits if hits > 0 else 0


def calculate_mean_average_precision(df):
    """
    Calculate the Mean Average Precision (MAP) for the given dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing columns 'query', 'ground_truth', 'top1', 'top5', 'top10'.

    Returns:
        float: The Mean Average Precision (MAP) score.
    """
    # Calculate AP for each query
    ap_scores = []
    for _, row in df.iterrows():
        ground_truth = row['ground_truth']
        predictions_with_scores = row['top10']
        predictions = [pred[0] for pred in predictions_with_scores]
        ap_scores.append(average_precision(ground_truth, predictions))

    # Calculate MAP
    mean_ap = sum(ap_scores) / len(ap_scores) if ap_scores else 0
    return mean_ap


def calculate_recall(df):
    """
    Calculate recall@1, recall@5, and recall@10 for the given dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing columns 'query', 'ground_truth', 'top1', 'top5', 'top10'.

    Returns:
        dict: A dictionary containing recall@1, recall@5, and recall@10.
    """
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    total_queries = len(df)

    for _, row in df.iterrows():
        ground_truth = row['ground_truth']
        if ground_truth in [pred[0] for pred in row['top1']]:
            recall_at_1 += 1
        if ground_truth in [pred[0] for pred in row['top5']]:
            recall_at_5 += 1
        if ground_truth in [pred[0] for pred in row['top10']]:
            recall_at_10 += 1

    return {
        'recall@1': recall_at_1 / total_queries,
        'recall@5': recall_at_5 / total_queries,
        'recall@10': recall_at_10 / total_queries
    }


def ndcg_score(ground_truth, predictions, k=10):
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) for a single query.

    Args:
        ground_truth (int): The ground truth video ID.
        predictions (list): List of predicted video IDs with scores [(id, score), ...].
        k (int): The number of top predictions to consider.

    Returns:
        float: The NDCG score for the query.
    """
    def dcg(relevance_scores):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

    # Relevance scores: 1 if the prediction matches the ground truth, else 0
    relevance_scores = [1 if pred[0] ==
                        ground_truth else 0 for pred in predictions[:k]]

    # Calculate DCG and IDCG
    actual_dcg = dcg(relevance_scores)
    ideal_dcg = dcg(sorted(relevance_scores, reverse=True))

    # Return NDCG
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

# call this function to get the NDCG score for each query


def calculate_ndcg(df, k=10):
    """
    Calculate NDCG for the given dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing columns 'query', 'ground_truth', 'top1', 'top5', 'top10'.
        k (int): The number of top predictions to consider.

    Returns:
        float: The mean NDCG score.
    """
    ndcg_scores = []
    for _, row in df.iterrows():
        ground_truth = row['ground_truth']
        predictions_with_scores = row['top10']
        ndcg_scores.append(ndcg_score(
            ground_truth, predictions_with_scores, k))

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0


def get_all_eval_scores(df):
    """Return a dataframe with all evaluation scores: Recall@1, Recall@5, Recall@10, MAP, NDCG@1, NDCG@5, NDCG@10"""
    recall_scores = calculate_recall(df)
    map_score = calculate_mean_average_precision(df)
    ndcg_score_1 = calculate_ndcg(df, k=1)
    ndcg_score_5 = calculate_ndcg(df, k=5)
    ndcg_score_10 = calculate_ndcg(df, k=10)

    eval_scores = {
        'recall@1': recall_scores['recall@1'],
        'recall@5': recall_scores['recall@5'],
        'recall@10': recall_scores['recall@10'],
        'map': map_score,
        'ndcg@1': ndcg_score_1,
        'ndcg@5': ndcg_score_5,
        'ndcg@10': ndcg_score_10
    }

    return eval_scores
