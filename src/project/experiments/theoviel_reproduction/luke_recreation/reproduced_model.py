import ast
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score
from typing import List, Union, Tuple

from project.data.data_loaders import get_clean_train_data
from sklearn.model_selection import GroupKFold

import pdb

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

    def __init__(self):
        self.train_data, self.valid_data, self.patient_notes = self.get_prepared_data()

    def get_prepared_data(self):
        data = get_clean_train_data()
        train_indices, test_indices = list(GroupKFold(n_splits=5).split(data['pn_history'], data['location'], data['pn_num']))[0]
        train_data = data.iloc[train_indices]
        valid_data = data.iloc[test_indices]
                
        train_data_grouped = train_data.groupby(['case_num', 'pn_num','pn_history'], as_index=False).agg(list)
        patient_notes = train_data_grouped

        patient_notes = patient_notes.dropna(axis=0).reset_index(drop=True)
        patient_notes = patient_notes[['case_num', 'pn_num', 'pn_history', 'annotation', 'location', 'feature_text', 'feature_num']]
        
        # Cache matching dict with keys (case_num, feature_num) and values (list of annotations) of entire corpus
        self.matching_dict = train_data.copy()[['case_num', 'feature_num', 'annotation']].groupby(['case_num', 'feature_num']).agg(list).T.to_dict()
        self.matching_dict = {k: np.concatenate(v['annotation']) for k, v in self.matching_dict.items()}
        self.matching_dict = {k: np.unique([v_.lower() for v_ in v]) for k, v in self.matching_dict.items()}

        return train_data, valid_data, patient_notes

    def location_to_span(location) -> List:
        '''
        Converts semicolon-delimited location string to list of spans.
        '''
        spans = []
        for loc in location:
            if ";" in loc:
                loc = loc.split(';')
            else:
                loc = [loc]
            
            for l in loc:
                spans.append(list(np.array(l.split(' ')).astype(int)))
        
        return spans

    def pred_to_location(self, pred: List) -> str:
        '''
        Converts predicted spans to string of proper "location" format with semi-colon delimiters in the patient note.

        ex: [[694, 704],[943,970]] -> "694 704;943 970"
        '''
        loc = ";".join([" ".join(np.array(p).astype(str)) for p in pred])

        return loc


    def predict(self, case_num: int, feature_num: int, patient_note: str) -> List[Tuple[int, int]]: # TODO: modify entire submission script to match this
        """
        Returns string of bounds if case number case_num and note patient_note has feature_num, returns '' otherwise.
        """
        candidates = self.matching_dict.get((case_num, feature_num), [])

        spans = []
        for c in candidates:
            start = patient_note.find(c)
            if start > -1:
                spans.append((start, start + len(c)))
    
        # list of predicted spans in their respective location of patient_note
        return spans


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