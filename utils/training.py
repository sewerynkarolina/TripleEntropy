import torch
from transformers.file_utils import is_tf_available, is_torch_available

import numpy as np
import random

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if is_tf_available():
        import tensorflow as tf
        tf.random.set_seed(seed)



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)


    return {
        'accuracy_score': acc,
        'f1_score': f1,
        'recall_score': recall,
        'precision_score': precision
    }
