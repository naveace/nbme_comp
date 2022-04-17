import ast
import numpy as np
import pandas as pd
from collections import Counter

from typing import List, Union, Tuple

from project.data.data_loaders import get_clean_train_data
from sklearn.model_selection import GroupKFold



def spans_to_binary(spans: List[List[int]], length=None) -> np.ndarray:
    """
    Converts spans to a binary array indicating whether each character is in the span.

    Args:
        spans (list of lists of two ints): Spans.

    Returns:
        np array [length]: Binarized spans.
    """
    length:int = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1
    return binary

def micro_f1(preds: List[List[int]], truths: List[List[int]]) -> float:
    """
    Micro f1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds)

class BaselineClassifier:
    """
        Implements Theo Viel's baseline string-matching classifier
    """
    def location_to_span(location):
        spans = []
        for loc in location:
            if ";" in loc:
                loc = loc.split(';')
            else:
                loc = [loc]
            
            for l in loc:
                spans.append(list(np.array(l.split(' ')).astype(int)))
        
        return spans

    def preds_to_location(preds):
        locations = []
        for pred in preds:
            loc = ";".join([" ".join(np.array(p).astype(str)) for p in pred])
            locations.append(loc)
        return locations


    def fit(self, X: pd.DataFrame, y: Union[np.ndarray, list]):
        pass

    def predict(self, X: str, Y: str) -> int:
        """
        Returns 1 if `X` has a 'Y' feature present in it and the feature's bounds, 0 otherwise
        """
        
        matching_dict = df_train[['case_num', 'feature_num', 'annotation']].groupby(['case_num', 'feature_num']).agg(list).T.to_dict()
        matching_dict = {k: np.concatenate(v['annotation']) for k, v in matching_dict.items()}
        matching_dict = {k: np.unique([v_.lower() for v_ in v]) for k, v in matching_dict.items()}

        preds = []
        for i in range(len(df_test)):
            key = (df_test['case_num'][i], df_test['feature_num'][i])

            candidates = matching_dict[key]

            text = df_test['pn_history'][i].lower()

            spans = []
            for c in candidates:
                start = text.find(c)
                if start > -1:
                    spans.append([start, start + len(c)])
            preds.append(spans)


if __name__ == "__main__":
    data = get_clean_train_data()
    train_indices, test_indices = list(GroupKFold(n_splits=5).split(data['pn_history'], data['location'], data['pn_num']))[0]
    train_data = data.iloc[train_indices]
    valid_data = data.iloc[test_indices]

    ####

    spans = patient_notes['location'][0]
    spans = [[list(np.array(s.split(' ')).astype(int)) for s in span] for span in spans if len(span)]

    pred = spans
    truth = [span[:2] for span in spans]

    print(pred)
    print(truth)