import pandas as pd


def twohee_data_col_to_df(twohee_data_collection):
    res_list = twohee_data_collection.to_list()
    res_obj_list = []
    for r in res_list:
        res_obj = vars(r)
        res_obj_list.append(res_obj)
    res_df = pd.DataFrame(res_obj_list)
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

    print(ap_scores)
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
