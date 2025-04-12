import pandas as pd
import json
from pprint import pprint

FIRE_MSRVTT_ANNOTATIONS = "data/FIRE/fire-data/fire_msrvtt_dataset.json"


def _open_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

# qrels should be a map that maps a query/queryid to a list of relevant documents that satisfies that query


def load_msrvtt_qrels():
    fire_annotations_raw = _open_json(FIRE_MSRVTT_ANNOTATIONS)
    annotations = fire_annotations_raw["annotations"]
    # unique_qs = set([a["query"] for a in annotations])
    annotations_df = pd.DataFrame(annotations)
    # groupby query
    grouped = annotations_df.groupby("query")
    qrels = {}
    for query, group in grouped:
        qrels[query] = list(group["relevant_ids"])
        # TODO FIX THISSS
    return qrels


def filter_single_result_queries(qrels_dict):
    return {k: v for k, v in qrels_dict.items() if len(v) == 1}


if __name__ == "__main__":
    qrels = load_msrvtt_qrels()
    print("Number of queries: ", len(qrels))
    print("Number of queries with single result: ",
          len(filter_single_result_queries(qrels)))
