import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
from .metrics import sensitivity, specificity

def evaluate_model(model, dataloader, device):
    predictions = []
    labels_arr = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']
            outputs = model(**batch)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            labels_arr.extend(labels.cpu().numpy())
    predictions = np.array(predictions)
    labels_arr = np.array(labels_arr)
    acc_ = accuracy_score(labels_arr, predictions)
    se_ = sensitivity(labels_arr, predictions)
    sp_ = specificity(labels_arr, predictions)
    return acc_, se_, sp_, predictions, labels_arr

def vizwiz_accuracy(df, true_ans, pred_ans):
    accuracy_sum = 0
    for _, row in df.iterrows():
        predicted_answer = row[pred_ans]
        answers_for_question = row[true_ans]
        correct_matches = sum(1 for answer in answers_for_question if answer == predicted_answer)
        accuracy_sum += min(1, correct_matches / 3)
    accuracy = accuracy_sum / len(df)
    return accuracy