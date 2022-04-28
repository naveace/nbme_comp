from project.experiments.theoviel_reproduction.luke_recreation.reproduced_model import *
import numpy as np
from project.data.data_loaders import get_clean_train_data
from typing import Final
TRAIN:Final = get_clean_train_data()

classifier = BaselineClassifier()

def test_empty_note_input():
    prediction = classifier.predict(0,0,"")
    assert type(prediction) == str


if __name__ == "__main__":
    test_empty_note_input()