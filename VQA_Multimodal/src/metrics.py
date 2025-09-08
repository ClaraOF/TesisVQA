import math
import re
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from config import DATASETS_PATH, DATA_ANS
from calculate_wups import wup_measure
from typing import Dict, List, Optional, Tuple

WORD = re.compile(r"\w+")

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

# def batch_cosine_measure(labels, preds, answer_space):
#     return np.mean([
#         get_cosine(text_to_vector(answer_space[label]), text_to_vector(answer_space[pred]))
#         for label, pred in zip(labels, preds)
#     ])
def batch_cosine_measure(labels, preds, answer_space):
    wup_scores = [wup_measure(answer_space[label], answer_space[pred]) for label, pred in zip(labels, preds)]
    return np.mean(wup_scores)

def batch_wup_measure(labels, preds, answer_space):
    return np.mean([
        wup_measure(answer_space[label], answer_space[pred])
        for label, pred in zip(labels, preds)
    ])

# def compute_metrics(eval_tuple, answer_space):
#     preds, labels = eval_tuple
#     return {
#         "cosine": batch_cosine_measure(labels, preds, answer_space),
#         "wups": batch_wup_measure(labels, preds, answer_space)
#     }
# Function to compute all relevant performance metrics, to be passed into the trainer
def compute_metrics(eval_tuple: Tuple[np.ndarray, np.ndarray], answer_space) -> Dict[str, float]:
    preds, labels = eval_tuple
    #preds = logits.argmax(axis=-1)
    return {
        #"acc": accuracy_score(labels, preds),
        "cosine": batch_cosine_measure(labels, preds,answer_space),
        "wups": batch_wup_measure(labels, preds,answer_space)
    }
def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro")

def specificity(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro")