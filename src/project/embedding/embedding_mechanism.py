import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


class CorpusEmbedder:
    """
    Allows for embedding any patient note from the patient_notes.csv file using either:
    - a https://www.sbert.net/ embedder, using the pretrained models from the table here: https://www.sbert.net/docs/pretrained_models.html
    - a hugging face embeddering, using the name given here https://huggingface.co/models?pipeline_tag=feature-extraction&sort=downloads
    - - e.g. 'openai/clip-vit-base-patch32'
    MODEL_NAMES is a class variable that contains the valid model names which can be passed to the constructor
    """
    MODEL_NAMES = set({"all-mpnet-base-v2"})
    OUTPUT_DIR = "./outputs"

    def __init__(self, model_name: str) -> None:
        """
        :param: model_name - the name of the model to use
        Should be one of the valid model names given by the MODEL_NAMES variable
        """
        self.model_name = model_name
        if model_name not in self.MODEL_NAMES:
            raise ValueError(f"Invalid model name {model_name}. Should be one of {self.MODEL_NAMES}")

    def embed(self, patient_row: pd.Series, cache=True) -> np.ndarray:
        """
        Returns a numpy array with the emebedding of shape (1, embedding_dim) for patient_row.
        If not returning from cache, embeds a row of patient_notes.csv.
        Optionally adds embedding to cache if not present already
        :param: patient_row - a row of patient_notes.csv
        :param: cache - whether to cache the embedding
        """
        embedding_output_path = f"{self.OUTPUT_DIR}/{self.model_name}" 
        # First create the cache directory if it doesn't exist
        if not os.path.exists(embedding_output_path):
            os.mkdir(embedding_output_path)

        # Check if the embedding for this patient row is in the cache
        embedding_name = f"{patient_row['patient_id']}.npy"
        if embedding_name not in os.listdir('.') and cache:
            embedder = SentenceTransformer(self.model_name)
            embedding = embedder.encode(patient_row['pn_history'])
            np.save(f"{embedding_output_path}/{embedding_name}", embedding)
        
        return np.load(f"{embedding_output_path}/{embedding_name}")
    

if __name__ == "__main__":
    embedder = CorpusEmbedder("all-mpnet-base-v2")
    embedding = embedder.embed(pd.Series({'patient_id': '1', 'pn_history': 'I am a patient with a history of diabetes'}))
    print(embedding.shape)