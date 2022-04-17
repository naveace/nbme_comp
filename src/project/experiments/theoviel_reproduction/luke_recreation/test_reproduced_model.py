import numpy as np
from project.data.data_loaders import get_clean_train_data
from typing import Final
from torch.utils.data import DataLoader
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


if __name__ == "__main__":
    test_empty_input()