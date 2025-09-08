import os
import pandas as pd
from datasets import load_dataset
from .config import DATASETS_PATH, DATA_TRAIN, DATA_VALI, DATA_ANS

def load_vqa_datasets():
    dataset = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(DATASETS_PATH, DATA_TRAIN),
            "test": os.path.join(DATASETS_PATH, DATA_VALI)
        }
    )
    with open(os.path.join(DATASETS_PATH, DATA_ANS)) as f:
        answer_space = f.read().splitlines()
    return dataset, answer_space

def add_label_column(dataset, answer_space):
    def label_mapper(examples):
        return {
            'label': [answer_space.index(ans.lower()) for ans in examples['answer_trad']]
        }
    dataset = dataset.map(label_mapper, batched=True)
    return dataset