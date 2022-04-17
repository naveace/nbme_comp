from project.hello_world import hello_world
import sys
from typing import List
import numpy as np
import pandas as pd
from os.path import join
from project.classifiers import MaleEvalClassifier, FemaleEvalClassifier
from tqdm import tqdm
hello_world()

class BaselineClassifier:
    def __init__(self) -> None:
        pass

class Classifiers:
    """
    Class to hold and serve all available classifiers for evaluation. If the requested classifier does not exist,
    it will be return the 'general' classifier, which reflects this project's baseline state of the art 
    (i.e. Theo Viel's naive 58% accurate baseline).

    """
    def __init__(self):
        CLASSIFIERS = {
            'male': MaleEvalClassifier(),
            'female': FemaleEvalClassifier(),
            'general': BaselineClassifier()
        }
        self.classifiers = CLASSIFIERS

    def __getitem__(self, index):
        if index not in self.classifiers:
            return self.classifiers['general']
        return self.classifiers[index]

        
def load_test_data() -> pd.DataFrame:
    """
    Returns a pandas DataFrame with all data relevant for evaluation. Each row has at minimum:
    - `id`: the id of the test pn_history x feature x case pair
    - `case_num`: the case number for the test sample
    - `pn_num`: the pn identifier for the test sample
    - `feature_text`: the text of the feature for the test sample
    - `feature_num`: the identifier of the feature for the test sample
    - `pn_history`: the text for the patient note of this test sample
    # TODO: avoid tokenizing patient notes more than once when we get to that point
    """
    DATA_PATH = 'data/' if len(sys.argv) == 1 else sys.argv[1]
    test = pd.read_csv(join(DATA_PATH, 'test.csv'))
    features = pd.read_csv(join(DATA_PATH, 'features.csv'))
    patient_notes = pd.read_csv(join(DATA_PATH, 'patient_notes.csv'))
    test = test.merge(features, on='feature_num').merge(patient_notes, on='pn_num')
    return test

def make_predictions(test: pd.DataFrame) -> List[str]:
    """
    Makes predictions using Male/Female classifiers for each item in test set
    Expects test set to be given by load_test_data()
    Inserts random predictions if no classifiers available
    """
    predictions = []
    classifiers = Classifiers()
    male_classifier, female_classifier = classifiers['male'], classifiers['female']
    for idx, r in tqdm(test.iterrows(), desc='Eval:', total=len(test)):
        if r['feature_text'] == 'Male':
            prediction = male_classifier.predict(r["pn_history"])
            bounds = f'{0} {len(r["pn_history"])}' if prediction == 1 else ''
            predictions.append(bounds)
        elif r['feature_text'] == 'Female':
            prediction = female_classifier.predict(r["pn_history"])
            bounds = f'{0} {len(r["pn_history"])}' if prediction == 1 else ''
            predictions.append(bounds)
        else:
            predictions.append(f'{np.random.randint(100)} {np.random.randint(101, 200)}')
    return predictions

if __name__ == '__main__':
    # Loading data
    test = load_test_data()
    # Making predictions
    predictions = make_predictions(test)
    test['location'] = predictions
    submission = test[['id', 'location']]
    submission.to_csv('submission.csv', index=False)