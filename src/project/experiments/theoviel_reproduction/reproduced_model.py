import os
from tqdm import tqdm
import pandas as pd
from transformers import BatchEncoding, AutoTokenizer, AutoModel, AutoConfig
from transformers.models.deberta.configuration_deberta import DebertaConfig
from transformers.models.deberta.tokenization_deberta_fast import DebertaTokenizerFast
from transformers.models.deberta.modeling_deberta import DebertaModel
import torch.nn as nn
import torch
from typing import List, Tuple
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from project.data.data_loaders import get_clean_train_data
from sklearn.metrics import f1_score

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
CACHE_DIR = os.environ.get('TRANSFORMERS_CACHE', f'{os.path.expanduser("~")}/.cache')
print(f'Using cache dir: {CACHE_DIR}, please set environment variable TRANSFORMERS_CACHE to override')
tokenizer:DebertaTokenizerFast = AutoTokenizer.from_pretrained(CFG.model, cache_dir=CACHE_DIR)
tokenizer.save_pretrained('tokenizer/')

def _encode_input(pn_history: str, feature_text: str, case_num: int) -> BatchEncoding:
    """
    Encodes the input text of a patient note as well as the feature text and case_num into a Batch Encoding
        which will be fed into the DeBerta model.
    Expected output of the model is a sequence of 1s and 0s for the tokens of pn_history
        where 1 indicates that feature `feature_text` is present in the token.
    :param pn_history: The patient note
    :param feature_text: The feature text
    :param case_num: The case number
    :return: The Batch Encoding of the pair of inputs with the following format:
        .input_ids: torch.tensor (CFG.max_len), sequence of tokens the input is mapped to, given by the id (int)
        .token_type_ids: torch.tensor (CFG.max_len), sequence of ids indicating what sentence a token belongs to (int)
        .attention_mask: torch.tensor (CFG.max_len), sequence of 1s and 0s indicating which tokens should be attended to (int)
    """
    context = f'Feature: {feature_text} Case: {case_num}'
    inputs:BatchEncoding = tokenizer(pn_history, context, add_special_tokens=True, max_length=CFG.max_len, padding='max_length', return_offsets_mapping=False)
    for key, value in inputs.items():
        inputs[key] = torch.tensor(value, dtype=torch.long)
    return inputs

def _create_label(text: str, location_list: List[Tuple[int, int]]) -> torch.Tensor:
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
    Wraps around a dataframe expected to have columns
        - pn_history: The patient note
        - feature_text: The feature text
        - location: The location pairs of the feature text in the patient note
    """
    def __init__(self, df: pd.DataFrame):
        self._TRAIN = df.copy(deep=True)

    def __len__(self):
        return len(self._TRAIN)

    def __getitem__(self, idx: int) -> Tuple[BatchEncoding, torch.Tensor]:
        row:pd.Series = self._TRAIN.iloc[idx]
        inputs = _encode_input(row['pn_history'], row['feature_text'], row['case_num'])
        label = _create_label(row['pn_history'], row['location'])
        return inputs, label

class DebertaCustomModel(nn.Module):
    """
    Represents a custom pre-trained DeBerta Model which can featurize BetchEncodings or produce outputs
    Outputs are scalars indicating presence (1) or non-presence (0) of the feature text in the patient note or whether a token is a special token (-1)
    """
    def __init__(self, path_to_saved_state_dict=''):
        super().__init__()
        self._setup_new_model()
        if path_to_saved_state_dict:
            print(f'Loading model from {path_to_saved_state_dict}')
            self.load_state_dict(torch.load(path_to_saved_state_dict, map_location=torch.device('cpu')))

    def _setup_new_model(self):
        """
        Sets up a brand new DeBerta model from pretrained using the model stored in CACHE_DIR if it exists
        """
        self._config:DebertaConfig = AutoConfig.from_pretrained(CFG.model, output_hidden_states=True, cache_dir=CACHE_DIR)
        self._model:DebertaModel = AutoModel.from_pretrained(CFG.model, config=self._config, cache_dir=CACHE_DIR)
        self._fc_dropout = nn.Dropout(CFG.fc_dropout)
        self._fc = nn.Linear(self._config.hidden_size, 1)
        self._init_weights(self._fc)
        
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initializes the weights of a module
        :param module: The module to initialize weights for
        TODO: See if this is necessary
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self._config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        else:
            raise Exception('Non-Linear modules not supported yet, see kaggle book for details')
        
    def feature(self, inputs: BatchEncoding) -> torch.Tensor:
        """
        Creates a feature embedding of the input
        :param inputs: The input to featurize
        :return: The feature embedding
        """
        outputs = self._model(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        """
        Produces output for the input. 
        If encoding is of size (batch x nTokens), output is of size (batch x nTokens x 1)
        """
        feature = self.feature(inputs)
        output = self._fc(self._fc_dropout(feature))
        return output
    
    def encoder_model(self) -> DebertaModel:
        """
        Returns the underlying DebertaModel, meant to be for reference only
        """
        return self._model
    
    def decoder_parameters(self) -> List[nn.Parameter]:
        """
        Returns the parameters of the decoder
        """
        return self._fc.parameters()
    
    def inference(self, text: str, feature: str, case_num: int) -> List[Tuple[int, int]]:
        """
        Performs inference on a patient note `text` and a feature `feature`
        :param text: The patient note
        :param feature: The feature text
        :return: The location pairs of the feature text in the patient note
        """
        assert torch.cuda.is_available(), 'Deberta model requires GPU'
        device = torch.device('cuda')
        self.to(device)
        self.eval()
        inputs = {k: v.to(device).view(1,-1) for k, v in _encode_input(text, feature, case_num).items()}
        with torch.no_grad():
            output = self.forward(inputs).sigmoid().cpu().numpy().reshape(-1) # we have only one piece of text, so should be able to merge into one vector
        return _get_predicted_character_level_bounds(output, text)




def get_optimizer(model: DebertaCustomModel) -> torch.optim.Optimizer:
    """
    Returns an optimizer for the model
    Applies 0 weight decay to bias and LayerNorm terms as well as the fully connected layer
    :param model: The model to optimize
    :return: The optimizer
    """
    is_decay_param = lambda name: not any(no_decay_indicator in name for no_decay_indicator in ["bias", "LayerNorm.bias", "LayerNorm.weight"])
    optimizer_parameters = [
        {'params': [param for name, param in model.encoder_model().named_parameters() if is_decay_param(name)],
            'lr': CFG.encoder_lr, 'weight_decay': CFG.weight_decay},
        {'params': [param for name, param in model.encoder_model().named_parameters() if not is_decay_param(name)],
            'lr': CFG.encoder_lr, 'weight_decay': 0.0},
        {'params': model.decoder_parameters(), 'lr': CFG.decoder_lr, 'weight_decay': 0.0}
    ]
    return torch.optim.AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)

def get_scheduler(optimizer: torch.optim.Optimizer, num_train_steps: int) -> object:
    """
    Given an optimizer returns a cosine lr scheduler wrapped around the optimizer
    """
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=CFG.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=CFG.num_cycles
    )
    return scheduler

def train_loop(model: DebertaCustomModel, train_loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: object, device: torch.device) -> DebertaCustomModel:
    """
    Trains the model for one epoch
    :param model: The model to train
    :param train_loader: The training data loader
    :param optimizer: The optimizer to use
    :param scheduler: The scheduler to use
    :param device: The device to use
    """
    model.train().to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    scalar = torch.cuda.amp.grad_scaler.GradScaler(enabled=True)
    losses = []
    for step, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        with torch.cuda.amp.autocast():
            y_preds = model.forward(inputs)
        loss:torch.Tensor = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()
        optimizer.zero_grad()
        scheduler.step()
        losses.append(loss.detach().cpu().item())
        if step % 100 == 0:
            print(f'Train loss: {np.mean(losses):.4f} ({np.std(losses):.4f})')
            losses.clear()
    return model

def val_loop(model: DebertaCustomModel, val_loader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray]:
    """
    Evaluates the model on the validation dataset and returns the validation loss and prediction on whole val set
        prexictions is (n_datapoints x n_tokens)
    :param model: The model to evaluate
    :param val_loader: The validation data loader
    :param device: The device to use
    :return: The validation loss and the predictions
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    losses = []
    predictions = []
    with torch.no_grad():
        for step, (inputs, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
            assert isinstance(inputs, dict) and isinstance(labels, torch.Tensor)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            y_preds = model.forward(inputs)
            loss:torch.Tensor = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
            loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
            losses.append(loss.detach().cpu().item())
            batch_size = labels.size(0)
            predictions.append(y_preds.sigmoid().cpu().numpy().reshape(batch_size, -1))
    return np.mean(losses), np.concatenate(predictions, axis=0)

def get_f1_score(predictions: np.ndarray, patient_notes: List[str], true_bounds: List[Tuple[int, int]]) -> float:
    """
    Evaluates the F1 score of a set of `predictions` on a set of `patient notes` given the `true_bounds`
    :param predictions: The predictions to evaluate, (n_datapoints x n_tokens)
    :param patient_notes: The patient notes to evaluate on (n_datapoints)
    :param true_bounds: The true bounds of the patient notes (n_datapoints)
    :return: The F1 score
    """
    predicted_bounds = [_get_predicted_character_level_bounds(model_prediction, raw_text) for 
                        model_prediction, raw_text in zip(predictions, patient_notes)]
    return _evaluate_f1(predicted_bounds, true_bounds)

def _get_pairs(lst: list) -> List[tuple]:
    """
    Returns inorder pairs from a list:
    e.g. [1,2,3,4] -> [(1,2), (2,3), (3,4)]
    Requires len(lst) >= 2
    """
    return list(zip(lst[:-1], lst[1:]))
def _get_idx_groups(char_idxs: List[int]) -> List[List[int]]:
    """
    Returns the groups of consecutive indices in a list
    e.g. [1,2,3,7,8,9] -> [[1,2,3], [7,8,9]]
    """
    if len(char_idxs) == 0:
        return []
    elif len(char_idxs) == 1:
        return [char_idxs]
    else:
        distance_from_next = np.array(list(map(lambda t: t[1] - t[0], _get_pairs(char_idxs))))
        split_list_idxs = [0] + list(np.where(distance_from_next > 1)[0] + 1) + [len(char_idxs)]
        groups = []
        for (start, end) in _get_pairs(split_list_idxs):
            groups.append(char_idxs[start:end])
        return groups
def _get_predicted_character_level_bounds(model_output: np.ndarray, raw_text: str) -> List[Tuple[int, int]]:
    """
    Returns the character level bounds of the predicted text
    :param model_output: The model output (n_datapoints x n_tokens)
    :param raw_text: The raw text of the patient (n_datapoints)
    :return: The character level bounds of the predicted text
    """
    char_level_text_prediction = np.zeros(len(raw_text))
    encoded_text = tokenizer(raw_text, add_special_tokens=True, return_offsets_mapping=True)
    for (start, end), pred in zip(encoded_text['offset_mapping'], model_output):
        char_level_text_prediction[start:end] = pred
    positive_char_idxs = np.where(char_level_text_prediction >= 0.5)[0]
    idx_groups = _get_idx_groups(positive_char_idxs)
    bounds = [(min(g), max(g)) for g in idx_groups]
    return bounds

def _spans_to_binary(spans: List[List[int]], length=None) -> np.ndarray:
    """
    Converts spans to a binary array indicating whether each character is in the span.

    Args:
        spans (list of lists of two ints): Spans.

    Returns:
        np array [length]: Binarized spans.
    """
    length:int = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1
    return binary
def _get_binarized_values(predicted_span: List[Tuple[int, int]], true_span: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the binarized values of the predicted and true spans
    :param predicted_span: The predicted span
    :param true_span: The true span
    :return: The binarized values of the predicted and true spans
    e.g. predicted = [(0,1), (2,3)], true = [(2,4)]
    Returns ([1,0,1,0,0], [0,0,1,1,1])
    """
    length = int(max(np.max(predicted_span, initial=0), np.max(true_span, initial=0)))
    return _spans_to_binary(predicted_span, length), _spans_to_binary(true_span, length)

def _evaluate_f1(predicted_bounds: List[Tuple[int, int]], true_locations:  List[Tuple[int, int]]) -> float:
    """
    Given a set of predicted bounds and true locations, returns the F1 score
    :param predicted_bounds: The predicted bounds
    :param true_locations: The true locations
    :return: The F1 score
    """
    binary_preds, binary_truths = [], []
    for predicted_bound, true_bound in zip(predicted_bounds, true_locations.values):
        pred_binary, true_binary = _get_binarized_values(predicted_bound, true_bound)
        binary_preds.append(pred_binary)
        binary_truths.append(true_binary)
    return f1_score(np.concatenate(binary_preds), np.concatenate(binary_truths))