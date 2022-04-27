import re
import nltk
import numpy as np
import pandas as pd
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, '../../data')
import data_loaders
# from project.data.data_loaders import get_clean_train_data, get_features


def phrase(text: str) -> list:
    """
    Divide patient notes into phrases by splitting by commas and periods and using nltk sent_tokenize.
    Gets a piece of text and returns a list of phrases
    """
    phrases = text.split(', ')
    phrases = [i.split('. ') for i in phrases]
    phrases = [sent_tokenize(val) for sublist in phrases for val in sublist]
    phrases = [val for sublist in phrases for val in sublist]

    return phrases


def process_feature_text(text: str) -> str:
    """
    Cleans feature text
    """
    text = re.sub('I-year', '1-year', text)
    text = re.sub('-OR-', " or ", text)
    text = re.sub('-', ' ', text)

    return text


def get_features(case=None):
    features = data_loaders.get_features()
    features['feature_text'] = features['feature_text'].apply(process_feature_text)
    if case is not None:
        features = features[features['case_num'] == case].reset_index(drop=True)
    return features


def train_bert(sentences: list, show_pbar=False):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(sentences, show_progress_bar=show_pbar)

    return embeddings, model


def bert_similarity(s: str, embeddings, model: SentenceTransformer) -> (int, float):
    """
    return the index and similarity score of the most similar feature to a given string
    """
    y = model.encode(s).reshape(1, -1)
    sim = cosine_similarity(embeddings, y)
    opt_index = np.argmax(sim)

    return opt_index, sim[opt_index][0]


def predict(feature_num: int, case_num:int, pn_note: str, threshold=0.47) -> list:
    """
    returns a list of tuples of predicted locations of a given feature in a given text
    """
    features = get_features(case_num)
    embeddings, model = train_bert(features['feature_text'])
    phrases = phrase(pn_note)
    start = 0
    locations = []
    similarities = []
    for phrase_txt in phrases:
        end = start + len(phrase_txt)
        index, sim = bert_similarity(phrase_txt, embeddings, model)
        f_num = features['feature_num'][index]
        if f_num == feature_num and sim > threshold:
            similarities.append(sim)
            locations.append((start, end))
        start += len(phrase_txt) + 2

    return locations


def prepare_test_data() -> pd.DataFrame:
    """
    get test data and adds the notes text
    """
    notes: pd.DataFrame = data_loaders.get_patient_notes()
    notes = notes.set_index('pn_num', drop=True)
    test = pd.read_csv('../../data/test.csv')
    test['pn_history'] = test['pn_num'].apply(lambda x: notes.iloc[x]['pn_history'])

    return test


test = prepare_test_data()
test['pred'] = test.apply(lambda row: predict(row['feature_num'],
                                              row['case_num'],
                                              row['pn_history'],
                                              threshold=0.37), axis=1)
