from project.experiments.theoviel_reproduction.reproduced_model import encode_input, create_label, TrainDataset, DebertaCustomModel
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

def seed_everything(seed=42):
    import random
    import os
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def test_model():
    seed_everything()
    kaggle_param_means = eval('{"model.embeddings.LayerNorm.bias": -0.01872188411653042, "model.encoder.layer.0.attention.self.q_bias": -0.013368003070354462, "model.encoder.layer.0.attention.self.v_bias": 0.008060838095843792, "model.encoder.layer.0.attention.self.pos_q_proj.bias": 0.001035042805597186, "model.encoder.layer.0.attention.output.dense.bias": -1.2946936749358429e-06, "model.encoder.layer.0.attention.output.LayerNorm.bias": -0.02889389358460903, "model.encoder.layer.0.intermediate.dense.bias": -0.07005833089351654, "model.encoder.layer.0.output.dense.bias": -0.0009392995852977037, "model.encoder.layer.0.output.LayerNorm.bias": 0.01381229143589735, "model.encoder.layer.1.attention.self.q_bias": -0.00044386088848114014, "model.encoder.layer.1.attention.self.v_bias": -0.00039144844049587846, "model.encoder.layer.1.attention.self.pos_q_proj.bias": -0.0016293670050799847, "model.encoder.layer.1.attention.output.dense.bias": -0.0002177990972995758, "model.encoder.layer.1.attention.output.LayerNorm.bias": -0.015322504565119743, "model.encoder.layer.1.intermediate.dense.bias": -0.0568847581744194, "model.encoder.layer.1.output.dense.bias": -0.0007762807654216886, "model.encoder.layer.1.output.LayerNorm.bias": 0.014861349016427994, "model.encoder.layer.2.attention.self.q_bias": 0.0016905500087887049, "model.encoder.layer.2.attention.self.v_bias": 0.00042778218630701303, "model.encoder.layer.2.attention.self.pos_q_proj.bias": 0.0010198315139859915, "model.encoder.layer.2.attention.output.dense.bias": 0.0002059343969449401, "model.encoder.layer.2.attention.output.LayerNorm.bias": -0.026901734992861748, "model.encoder.layer.2.intermediate.dense.bias": -0.05724099278450012, "model.encoder.layer.2.output.dense.bias": 6.030313670635223e-06, "model.encoder.layer.2.output.LayerNorm.bias": 0.02129512093961239, "model.encoder.layer.3.attention.self.q_bias": 0.003707442432641983, "model.encoder.layer.3.attention.self.v_bias": -0.0008457490475848317, "model.encoder.layer.3.attention.self.pos_q_proj.bias": -0.0003796924720518291, "model.encoder.layer.3.attention.output.dense.bias": -9.288088040193543e-05, "model.encoder.layer.3.attention.output.LayerNorm.bias": -0.026793038472533226, "model.encoder.layer.3.intermediate.dense.bias": -0.05537831038236618, "model.encoder.layer.3.output.dense.bias": -3.362509232829325e-05, "model.encoder.layer.3.output.LayerNorm.bias": -0.012742104008793831, "model.encoder.layer.4.attention.self.q_bias": -0.0033418573439121246, "model.encoder.layer.4.attention.self.v_bias": -0.0003224016400054097, "model.encoder.layer.4.attention.self.pos_q_proj.bias": -0.0043077110312879086, "model.encoder.layer.4.attention.output.dense.bias": 8.928349416237324e-05, "model.encoder.layer.4.attention.output.LayerNorm.bias": -0.027524104341864586, "model.encoder.layer.4.intermediate.dense.bias": -0.057714954018592834, "model.encoder.layer.4.output.dense.bias": 2.934240546892397e-05, "model.encoder.layer.4.output.LayerNorm.bias": -0.009261813014745712, "model.encoder.layer.5.attention.self.q_bias": 0.0030004573054611683, "model.encoder.layer.5.attention.self.v_bias": -0.0009373065549880266, "model.encoder.layer.5.attention.self.pos_q_proj.bias": -0.0009702934767119586, "model.encoder.layer.5.attention.output.dense.bias": 9.275988850276917e-05, "model.encoder.layer.5.attention.output.LayerNorm.bias": -0.005213789641857147, "model.encoder.layer.5.intermediate.dense.bias": -0.049343716353178024, "model.encoder.layer.5.output.dense.bias": 0.0002599772997200489, "model.encoder.layer.5.output.LayerNorm.bias": -0.012019680812954903, "model.encoder.layer.6.attention.self.q_bias": -0.00064975576242432, "model.encoder.layer.6.attention.self.v_bias": 0.00037524255458265543, "model.encoder.layer.6.attention.self.pos_q_proj.bias": 0.0006454527610912919, "model.encoder.layer.6.attention.output.dense.bias": 0.0003099963360000402, "model.encoder.layer.6.attention.output.LayerNorm.bias": 0.003626746591180563, "model.encoder.layer.6.intermediate.dense.bias": -0.05303549766540527, "model.encoder.layer.6.output.dense.bias": 7.382555486401543e-05, "model.encoder.layer.6.output.LayerNorm.bias": -0.02015323005616665, "model.encoder.layer.7.attention.self.q_bias": -0.002802572911605239, "model.encoder.layer.7.attention.self.v_bias": -0.00010490739805391058, "model.encoder.layer.7.attention.self.pos_q_proj.bias": -0.0026522446423768997, "model.encoder.layer.7.attention.output.dense.bias": 9.735161438584328e-05, "model.encoder.layer.7.attention.output.LayerNorm.bias": 0.023847665637731552, "model.encoder.layer.7.intermediate.dense.bias": -0.05433777719736099, "model.encoder.layer.7.output.dense.bias": 0.00037417569546960294, "model.encoder.layer.7.output.LayerNorm.bias": -0.020023107528686523, "model.encoder.layer.8.attention.self.q_bias": -0.00041228625923395157, "model.encoder.layer.8.attention.self.v_bias": -0.00020404811948537827, "model.encoder.layer.8.attention.self.pos_q_proj.bias": -0.0007955966866575181, "model.encoder.layer.8.attention.output.dense.bias": -0.0002775319735519588, "model.encoder.layer.8.attention.output.LayerNorm.bias": 0.029635585844516754, "model.encoder.layer.8.intermediate.dense.bias": -0.05047871172428131, "model.encoder.layer.8.output.dense.bias": -5.75150697841309e-05, "model.encoder.layer.8.output.LayerNorm.bias": -0.01842450350522995, "model.encoder.layer.9.attention.self.q_bias": -0.004358930978924036, "model.encoder.layer.9.attention.self.v_bias": 0.00048549420898780227, "model.encoder.layer.9.attention.self.pos_q_proj.bias": -0.0010180545505136251, "model.encoder.layer.9.attention.output.dense.bias": -0.000227226410061121, "model.encoder.layer.9.attention.output.LayerNorm.bias": 0.032152898609638214, "model.encoder.layer.9.intermediate.dense.bias": -0.046192318201065063, "model.encoder.layer.9.output.dense.bias": 8.74128527357243e-05, "model.encoder.layer.9.output.LayerNorm.bias": -0.01743406057357788, "model.encoder.layer.10.attention.self.q_bias": -0.0007120752707123756, "model.encoder.layer.10.attention.self.v_bias": -0.00033491122303530574, "model.encoder.layer.10.attention.self.pos_q_proj.bias": 0.0024776621721684933, "model.encoder.layer.10.attention.output.dense.bias": -0.00011856978380819783, "model.encoder.layer.10.attention.output.LayerNorm.bias": 0.03847266733646393, "model.encoder.layer.10.intermediate.dense.bias": -0.03498169779777527, "model.encoder.layer.10.output.dense.bias": 0.00018997280858457088, "model.encoder.layer.10.output.LayerNorm.bias": -0.016387851908802986, "model.encoder.layer.11.attention.self.q_bias": 0.0033513670787215233, "model.encoder.layer.11.attention.self.v_bias": -0.0020278978627175093, "model.encoder.layer.11.attention.self.pos_q_proj.bias": -0.0006971051916480064, "model.encoder.layer.11.attention.output.dense.bias": -0.001346013741567731, "model.encoder.layer.11.attention.output.LayerNorm.bias": 0.0744047462940216, "model.encoder.layer.11.intermediate.dense.bias": -0.02657308802008629, "model.encoder.layer.11.output.dense.bias": -2.0957164451829158e-05, "model.encoder.layer.11.output.LayerNorm.bias": 0.025717567652463913, "fc.bias": 0.0}')
    model = DebertaCustomModel().train()
    this_model_param_means = {k: v.float().mean().item() for k, v in model.state_dict().items() if (k.endswith('weights') or k.endswith('bias'))}
    for k, v in this_model_param_means.items():
        kaggle_key = '.'.join(['model' if not k.startswith('_fc') else 'fc'] + k.split('.')[1:])  # whatever we call our model internally should not matter, called model on kaggle
        assert np.isclose(v, kaggle_param_means[kaggle_key]), f'{kaggle_key} differs. Ours: {v:.5f}, theirs: {kaggle_param_means[kaggle_key]:.5f}'
    assert model.deberta_model()