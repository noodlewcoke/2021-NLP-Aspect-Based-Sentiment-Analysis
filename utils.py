import sys
import numpy as np
import torch


def bio_weights(trainset):

    labels = trainset[:len(trainset)]['label'].view(-1)
    n_b = torch.sum((labels==2))
    n_i = torch.sum((labels==1))
    n_o = torch.sum((labels==0))
    n = n_b + n_i + n_o

    b_weight = 1 - n_b / n
    i_weight = 1 - n_i / n
    o_weight = 1 - n_o / n

    return b_weight, i_weight, o_weight

def sentiment_weights(trainset):
    sentiments = trainset[:len(trainset)]['aspect_polarities'].view(-1)
    n_0 = torch.sum((sentiments == 0))
    n_1 = torch.sum((sentiments == 1))
    n_2 = torch.sum((sentiments == 2))
    n_3 = torch.sum((sentiments == 3))
    n = n_1 + n_2 + n_0 + n_3
    n0_weight = 1 - (n_0 / n)
    n1_weight = 1 - (n_1 / n)
    n2_weight = 1 - (n_2 / n)
    n3_weight = 1 - (n_3 / n)

    print('Sentiment class weights:\n')
    print(n0_weight, n1_weight, n2_weight, n3_weight)
    return n0_weight, n1_weight, n2_weight, n3_weight


def cleanse_predictions(predictions, labels, padding_idx):
    c_pred, c_label = [], []
    predictions = predictions.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()

    for pred, label in zip(predictions, labels):
        if not int(label) == padding_idx:
            c_pred.append(pred)
            c_label.append(label)
    assert len(c_pred) != len(list(predictions)), f"\n {list(labels)}"
    return c_pred, c_label

def index_based_accuracy(sentences, predictions, labels, crf=False):
    sentences = sentences.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()
    pred_targets, gold_targets = [], []

    padding_indices = np.where(sentences == 0)[0]
    sentences = np.delete(sentences, padding_indices)
    if not crf:
        predictions = predictions.view(-1).cpu().numpy()
        predictions = np.delete(predictions, padding_indices)
    else:
        predictions = [i for j in predictions for i in j]
    labels = np.delete(labels, padding_indices)

    for t, tmp in enumerate([predictions, labels]):
        prev = '_'
        targets = []
        target = []
        it = 0
        for i, tag in enumerate(tmp):
            if prev == 2:
                if tag == 2:
                    targets.append(tuple(target))
                    target = [(i, 2)]

                elif tag == 1:
                    target.append((i, 1))
                    prev = 1
                elif tag == 0:
                    targets.append(tuple(target))
                    target = []
                    prev = '_'
            elif prev == 1:
                if tag == 2:
                    targets.append(tuple(target))
                    target = [(i, 2)]
                    prev = 2
                elif tag == 1:
                    target.append((i, 1))
                    prev = 1
                elif tag == 0:
                    targets.append(tuple(target))
                    target = []
                    prev = '_'
            elif prev == '_':
                if tag == 2:
                    target.append((i, 2))
                    prev = 2
            it+=1
        if t==0:
            pred_targets = targets
            it = 0
        else:
            gold_targets = targets
    gold_targets = set(gold_targets)
    pred_targets = set(pred_targets)

    tp = len(pred_targets & gold_targets)
    fp = len(pred_targets - gold_targets)
    fn = len(gold_targets - pred_targets)
    precision, recall, f1 = 0, 0, 0
    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1