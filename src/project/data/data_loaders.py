import pandas as pd
import os
from functools import reduce
from typing import List, Tuple
import json
CWD = os.path.dirname(__file__)
def get_features() -> pd.DataFrame:
    """
    Returns a dataframe of the features from NBME
    """
    return pd.read_csv(f'{CWD}/features.csv')

def get_patient_notes() -> pd.DataFrame:
    """
    Returns a dataframe of the patient notes
    """
    return pd.read_csv(f'{CWD}/patient_notes.csv')

def get_train() -> pd.DataFrame:
    """
    Returns a dataframe of the training data
    """
    return pd.read_csv(f'{CWD}/train.csv')

def get_clean_train_data() -> pd.DataFrame:
    """
    Produces a cleaned version of the train dataframe with the following:
    id: a unique id for the train instance (as in Kaggle)
    case_num: unique id for case (as in Kaggle)
    pn_num: unique id for patient note (as in Kaggle)
    feature_num: unique id for patient note (as in kaggle)
    annotation: a List[str] of the text in patient note pn_num that corresponds to feature pn_num. 
        Each element is not necessarily contiguous but is human-readable. Some elements may splice together parts of patient note
    location_raw: the raw location (as in Kaggle)
    feature_text: the feature text for the feature feature_num (as in Kaggle)
    pn_history: the patient note (as in Kaggle)
    location: a List[Tuple[start: int, end: int]] of the ranges of characters in the patient note that corresponds to the feature (positives)
    """
    train, features, patient_notes = get_train(), get_features(), get_patient_notes()
    features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago" # This feature is incorrectly coded as 'Last-Pap-smear-I-year-ago'
    train = train.merge(features, on=['feature_num', 'case_num'], how='left').merge(patient_notes, on=['pn_num', 'case_num'], how='left')
    # Corrections taken from https://www.kaggle.com/code/naveace/nbme-deberta-base-baseline-train/edit
    for idx, correction_dict in json.load(open(f'{CWD}/annotation_corrections.json', 'r')).items():
        idx = int(idx)
        train.loc[idx, 'annotation'] = correction_dict['correct annotation']
        train.loc[idx, 'location'] = correction_dict['correct location']
    # Creating cleaned `location` column
    train['annotation'] = train['annotation'].map(eval)
    train['location'] = train['location'].map(eval)
    train['location_idx_tuple_pairs'] = train['location'].map(_convert_to_idx_tuple_pairs)
    assert_correctness(train['location_idx_tuple_pairs'])
    train = train.rename(columns={'location': 'location_raw', 'location_idx_tuple_pairs': 'location'})
    return train

def _get_start_and_end_indices(locations: str) -> List[Tuple[int, int]]:
    """
    Maps strings of the form '\d+ \d+(;\d+ \d+)*' to a list of tuples of the form (start, end).
    E.g. '1 2' -> [(1, 2)]
         '1 2;3 4' -> [(1, 2), (3, 4)]
         '' -> []
    """
    assert(isinstance(locations, str))
    if locations == '':
        return []
    if not ';' in locations:
        s, e = locations.split()
        return [(int(s), int(e))]
    return list(map(lambda s: _get_start_and_end_indices(s)[0], locations.split(';')))

def _convert_to_idx_tuple_pairs(location_list: List[str]) -> List[Tuple[int, int]]:
    """
    Maps a list of strings of the form '\d+ \d+(;\d+ \d+)*' to a list of tuples of the form (start, end).
    E.g. ['1 2', '3 4'] -> [(1, 2), (3, 4)]
    """
    return list(reduce(lambda l1, l2: l1 + l2, map(_get_start_and_end_indices, location_list), []))
def assert_correctness(labels) -> None:
    """
    Asserts correct types for labels
    """
    for idx, element in enumerate(labels):
        try:
            assert(isinstance(element, list))
            for obj in element:
                assert(isinstance(obj, tuple))
                assert(len(obj) == 2)
                assert(isinstance(obj[0], int))
                assert(isinstance(obj[1], int))
        except AssertionError:
            print(f'Failed assertion for element {idx}: {element}')

def get_synonym_training_data() -> pd.DataFrame:
    """
    Unique from get_clean_train_data() in the following way:
    - Contains 6 copies of a row, 1 original and 5 with  with one word in pn_history replaced with a synonym
    _________________________________________________________________________
    Produces a cleaned version of the train dataframe with the following:
    id: a unique id for the train instance (as in Kaggle)
    case_num: unique id for case (as in Kaggle)
    pn_num: unique id for patient note (as in Kaggle)
    feature_num: unique id for patient note (as in kaggle)
    annotation: a List[str] of the text in patient note pn_num that corresponds to feature pn_num. 
        Each element is not necessarily contiguous but is human-readable. Some elements may splice together parts of patient note
    location_raw: the raw location (as in Kaggle)
    feature_text: the feature text for the feature feature_num (as in Kaggle)
    pn_history: the patient note (as in Kaggle)
    location: a List[Tuple[start: int, end: int]] of the ranges of characters in the patient note that corresponds to the feature (positives)
    """
    df =  pd.read_csv(f'{CWD}/train_data_with_synonyms.csv', index_col=0)
    df['annotation'] = df['annotation'].map(eval)
    df['location'] = df['location'].map(eval)
    return df
