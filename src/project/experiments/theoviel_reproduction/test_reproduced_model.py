from venv import create
from project.experiments.theoviel_reproduction.reproduced_model import encode_input, create_label, TrainDataset
import numpy as np
from project.data.data_loaders import get_clean_train_data
from typing import Final
TRAIN:Final = get_clean_train_data()

def test_empty_input():
    encoding = encode_input("", "")
    input_ids:np.ndarray = encoding.input_ids.numpy()
    token_type_ids:np.ndarray = encoding.token_type_ids.numpy()
    attention_mask:np.ndarray = encoding.attention_mask.numpy()
    assert np.all(input_ids[0:2] == [1,2])
    assert np.all(input_ids[2:] == 0)
    assert np.all(token_type_ids == 0)
    assert np.all(attention_mask[0:2] == [1,1])
    assert np.all(attention_mask[2:] == 0)

def test_patient_note_and_feature():
    HISTORY = TRAIN.pn_history.values[0]
    FEATURE = TRAIN.feature_text.values[0]
    encoding = encode_input(HISTORY, FEATURE)
    input_ids:np.ndarray = encoding.input_ids.numpy()
    token_type_ids:np.ndarray = encoding.token_type_ids.numpy()
    attention_mask:np.ndarray = encoding.attention_mask.numpy()
    assert len(np.nonzero(input_ids)[0]) == 270
    assert input_ids.max() == 50121
    assert input_ids[input_ids.nonzero()[0]].min() == 1
    assert np.all(token_type_ids == 0)
    assert np.all(attention_mask[0:2] == [1,1])
    assert len(np.nonzero(attention_mask)[0]) == 270

def test_create_label_single_loc():
    label:np.ndarray = create_label(TRAIN.loc[20, 'pn_history'], TRAIN.loc[20, 'location']).numpy()
    assert label.shape == (466,)
    assert label[0] == -1
    assert np.all(np.where(label == 1)[0] == [101, 102])


def test_create_label_multi_loc():
    label:np.ndarray = create_label(TRAIN.loc[3, 'pn_history'], TRAIN.loc[3, 'location']).numpy()
    assert label.shape == (466,)
    assert label[0] == -1
    assert np.all(np.where(label == 1)[0] == [20, 21, 43])

def test_train_dataset():
    data = TrainDataset(TRAIN)
    assert len(data) == len(TRAIN)
    (test_X, test_y) = data[42]
    assert test_X.input_ids[0].item() == 1
    assert test_y[39].item() == 1

