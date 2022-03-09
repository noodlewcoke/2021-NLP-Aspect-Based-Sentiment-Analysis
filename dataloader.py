import numpy as np
import json, os, sys, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

os.chdir(os.path.dirname(os.path.realpath(__file__)))


class absaDataset(Dataset):
    
    def __init__(self, filepath, mode='ae'):
        with open(filepath, 'r') as reader:
            data = json.load(reader)
            reader.close()
        self.mode = mode

        self.sentences = data['sentences']
        self.masks = data['masks']
        self.labels = data['ae_labels']
        self.crf_labels = data['extra_labels']
        if self.mode == 'asc':
            self.target_indices = data['target_indices']
            self.aspect_polarities = data['aspect_polarities']
        del data
        self.data_len = len(self.sentences)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mode == 'asc':
            return {'sentence' : torch.tensor(self.sentences[idx]),
                    'mask' : torch.tensor(self.masks[idx]),
                    'label' : torch.tensor(self.labels[idx]),
                    'crf_label' : torch.tensor(self.crf_labels[idx]),
                    'target_indices' : torch.tensor(self.target_indices[idx]),
                    'aspect_polarities' : torch.tensor(self.aspect_polarities[idx])
            }
        else:
            return {'sentence' : torch.tensor(self.sentences[idx]),
                    'mask' : torch.tensor(self.masks[idx]),
                    'label' : torch.tensor(self.labels[idx]),
                    'crf_label' : torch.tensor(self.crf_labels[idx])
            }
        


if __name__ == '__main__':
    filepath = 'data/tokenized_laptops_train.json'
    dataset = absaDataset(filepath)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    for batch in dataloader:
        sentences = batch['sentence']
        masks = batch['mask']
        labels = batch['label']

        print(sentences.size())
        print(masks.size())
        print(labels.size())
        label = labels.view(-1).tolist()
        print(label)
        label = label[:label.index(3)]
        print(label)
        exit()