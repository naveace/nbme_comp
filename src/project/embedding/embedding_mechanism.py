import pandas as pd
import numpy as np
from typing import Set


class CorpusEmbedder:
    """
    Embeds a patient note from patient_notes.csv using a Hugging Face embedding model
    MODEL_NAMES is a class variable that contains the valid model names which can be passed to the constructor
    """
    MODEL_NAMES = set()

    def __init__(self, model_name: str) -> None:
        """
        :param: model_name - the name of the Hugging Face model to use
        Should be one of the valid model names given by the MODEL_NAMES variable
        """
        pass


    def embed(self, patient_row: pd.Series, cache=True) -> np.ndarray:
        """
        Emebds a row of patient_notes.csv; returns a numpy array with the emebedding of shape (1, embedding_dim)
        Optionally adds embedding to cache if not present already
        :param: patient_row - a row of patient_notes.csv
        :param: cache - whether to cache the embedding
        """
        pass

    
