from transformers import BatchEncoding, AutoTokenizer
import torch

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
    max_len=512
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=True
tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.save_pretrained('tokenizer/')
CFG.tokenizer = tokenizer


def encode_input(pn_history: str, feature_text: str) -> BatchEncoding:
    """
    Encodes the input text of a patient note as well as the feature text into a Batch Encoding
        which will be fed into the DeBerta model.
    Expected output of the model is a sequence of 1s and 0s for the tokens of pn_history
        where 1 indicates that feature `feature_text` is present in the token.
    :param pn_history: The patient note
    :param feature_text: The feature text
    :return: The Batch Encoding of the pair of inputs
    """
    return prepare_input(CFG, pn_history, feature_text)


def prepare_input(cfg, text, feature_text):
    """
    Copied verbatim from notebook. TODO: refactor out
    """
    inputs = cfg.tokenizer(text, feature_text, 
                           add_special_tokens=True,
                           max_length=CFG.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs