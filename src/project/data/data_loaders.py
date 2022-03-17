import pandas as pd
import os
from functools import reduce
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
