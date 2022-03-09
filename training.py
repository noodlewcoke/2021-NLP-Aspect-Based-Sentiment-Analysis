import numpy as np
import json, os, sys, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from transformers import BertModel, BertTokenizer, BertLayer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from TorchCRF import CRF
from torch.utils.data import DataLoader
from transformers.models import bert
from dataloader import absaDataset
from ae_models import LitAspectExtractorSimple, LitAspectExtractorBetter
from asc_models import myLitASC, myLitASC_Multi
from utils import bio_weights, sentiment_weights


bert_config = {
  "_name_or_path": "bert-base-cased",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": False,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.2.2",
  "type_vocab_size": 2,
  "use_cache": True,
  "vocab_size": 28996
}

hyperparameters = {
  'bert_lr' : 5e-5,
  'hidden_lr' : 1e-3,
  'lr_gamma' : 0.8,
  'bert_dropout' : 0.1,
  'hidden_dropout' : 0.7,
  'step_size' : 3,
  'bias' : True
}


def ae_training():
    global bert_config, hyperparameters
    filepath = 'data/tokenized_merged_ae_train.json'

    trainset = absaDataset(filepath)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    filepath = 'data/tokenized_merged_ae_dev.json'

    devset = absaDataset(filepath)
    devloader = DataLoader(devset, batch_size=32, shuffle=False, num_workers=4)
    tag_weights = bio_weights(trainset)

    config = [bert_config, hyperparameters]
    model = LitAspectExtractorBetter(config, tag_weights, mode='h')
    # model = LitAspectExtractorSimple(bert_config)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
      monitor='val/epoch_f1',
      mode = 'max'
    )

    trainer = pl.Trainer(gpus=1, 
                         auto_lr_find = False, 
                         max_epochs=30, 
                         fast_dev_run=False,
                         callbacks=[lr_monitor, checkpoint_callback])

    trainer.fit(model, trainloader, devloader)

def asc_training():
    global bert_config, hyperparameters
    filepath = 'data/tokenized_merged_asc_train.json'

    trainset = absaDataset(filepath, 'asc')
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    filepath = 'data/tokenized_merged_asc_dev.json'
    devset = absaDataset(filepath, 'asc')
    devloader = DataLoader(devset, batch_size=32, shuffle=False, num_workers=4)

    label_weights = sentiment_weights(trainset)
    config = [bert_config, hyperparameters]
    # model = myLitASC(config, label_weights=None)
    model = myLitASC_Multi(config, label_weights=None)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    checkpoint_callback = ModelCheckpoint(
      monitor='val/epoch_f1',
      mode = 'max'
    )


    trainer = pl.Trainer(gpus=1, 
                         auto_lr_find = False, 
                         max_epochs=30, 
                         fast_dev_run=False,
                         callbacks=[lr_monitor, checkpoint_callback])

    trainer.fit(model, trainloader, devloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help='ae or asc')
    args = parser.parse_args()

    if args.model.lower() == 'ae':
        print("AE TRAINING\n")
        ae_training()
    elif args.model.lower() == 'asc':
        print('ASC TRAINING\n')
        asc_training()
    else:
        print('Choose either "ae" or "asc"')
        exit()
