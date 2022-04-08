from project.experiments.theoviel_reproduction.reproduced_model import encode_input
import numpy as np
from project.data.data_loaders import get_clean_train_data

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
    train = get_clean_train_data()
    HISTORY = train.pn_history.values[0]
    FEATURE = train.feature_text.values[0]
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