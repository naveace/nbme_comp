import pandas as pd
from transformers import BatchEncoding, AutoTokenizer
import torch
from typing import List, Tuple
import numpy as np
from torch.utils.data import Dataset
from project.data.data_loaders import get_clean_train_data
# Copied verbatim from notebook. TODO: refactor out
class CFG:
    wandb=False
    competition='NBME'
    _wandb_kernel='nakama'
    debug=False
    apex=True
    print_freq=100
    num_workers=4
    model="microsoft/deberta-base"
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=5
    encoder_lr=2e-5
    decoder_lr=2e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=12
    fc_dropout=0.2
    max_len=466  # NOTE: This gets changed in the middle of notebook from 512 to 466
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=True
tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.save_pretrained('tokenizer/')

def encode_input(pn_history: str, feature_text: str) -> BatchEncoding:
    """
    Encodes the input text of a patient note as well as the feature text into a Batch Encoding
        which will be fed into the DeBerta model.
    Expected output of the model is a sequence of 1s and 0s for the tokens of pn_history
        where 1 indicates that feature `feature_text` is present in the token.
    :param pn_history: The patient note
    :param feature_text: The feature text
    :return: The Batch Encoding of the pair of inputs with the following format:
        .input_ids: torch.tensor (CFG.max_len), sequence of tokens the input is mapped to, given by the id (int)
        .token_type_ids: torch.tensor (CFG.max_len), sequence of ids indicating what sentence a token belongs to (int)
        .attention_mask: torch.tensor (CFG.max_len), sequence of 1s and 0s indicating which tokens should be attended to (int)
    """
    inputs:BatchEncoding = tokenizer(pn_history, feature_text, add_special_tokens=True, max_length=CFG.max_len, padding='max_length', return_offsets_mapping=False)
    for key, value in inputs.items():
        inputs[key] = torch.tensor(value, dtype=torch.long)
    return inputs

def create_label(text: str, location_list: List[Tuple[int, int]]) -> torch.Tensor:
    """
    Creates a label for the tokens correpsonding to input_text given a list of start and end locations (in the input text) of characters that should be given label 1
    Label gives -1 to special tokens, 0 to tokens with no characters they correspond to in location_list, and 1 to all other tokens
    Ex. 'Hello, how are you', [(0, 4), (15,17)] tokenized as ['[CLS]', 'Hello','[SEP]' , 'how','[SEP]', 'are','[SEP]', 'you'] receives label [-1, 1, -1, 0, -1, 0 -1, 1]
    :param text: The input text
    :param location_list: The list of start and end locations of characters that should be given label 1
    :return: The label for the tokens corresponding to input_text 
    """
    encoded:BatchEncoding = tokenizer(text, add_special_tokens=True, max_length=CFG.max_len, padding="max_length", return_offsets_mapping=True)
    offset_mapping:List[Tuple[int, int]] = encoded['offset_mapping'] # Maps back to the original text (e.g. 'Hello' -> 'He', 'llo', [(0, 2), (2, 5)])
    ignore_idxes:np.ndarray = np.where(np.array(encoded.sequence_ids()) != 0)[0]  # Indexes of the special tokens, see https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/tokenizer#transformers.BatchEncoding
    label = np.zeros(len(offset_mapping)) # Create a label for each token
    label[ignore_idxes] = -1 # Special tokens are given label -1 by default
    if not location_list: return torch.tensor(label, dtype=torch.float)  # If no locations are given, return the label
    for label_start, label_end in location_list:
        start_idx:int = None
        end_idx:int = None
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            if (start_idx is None) and (label_start < token_start):
                start_idx = idx - 1 # Give start the first token before the first token contained in [start, end]
            if (end_idx is None) and (label_end <= token_end):
                end_idx = idx + 1 # Give end the first token after the last token contained in [start, end]
        if start_idx is None:
            start_idx = end_idx
        if (start_idx != -1) and (end_idx != -1):
            label[start_idx:end_idx] = 1 # Give a token label of 1 if the idxs it corresponds to are within (start, end)
    return torch.tensor(label, dtype=torch.float)


class TrainDataset(Dataset):
    """
    Represents and immutable training dataset.
    """
    def __init__(self, df: pd.DataFrame):
        self._TRAIN = df.copy(deep=True)

    def __len__(self):
        return len(self._TRAIN)

    def __getitem__(self, idx: int) -> Tuple[BatchEncoding, torch.Tensor]:
        row:pd.Series = self._TRAIN.iloc[idx]
        inputs = encode_input(row['pn_history'], row['feature_text'])
        label = create_label(row['pn_history'], row['location'])
        return inputs, label

"""
Next steps: 
- Add in model class, ensure it is compatible with Train Dataset. Note that it takes a *significant* amount of RAM just to run the model once. 
- Addi in train loop and get this thing training!


"""