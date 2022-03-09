import numpy as np
import json, os, sys, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from transformers import BertModel, BertTokenizer, BertLayer, BertConfig
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from utils import cleanse_predictions, index_based_accuracy
from TorchCRF import CRF
from ae_models import ptMeta

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
os.chdir(os.path.dirname(os.path.realpath(__file__)))



# ASPECT SENTIMENT CLASSIFICATION

class myASC(ptMeta):
    '''
    My own approach inspired from "IMPROVING BERT PERFORMANCE FOR ASPECT-BASED SENTIMENT ANALYSIS".
    In their approach they extract the sentiment without considering the target word.
    Thus, if there are more than one aspect sentiments in the sentences their model would fail to detect all of them.
    Instead, I use the pooling layer and use it not only on the [CLS] token but on the target tokens as well.
    Afterwards aggregating them resulting representations, and propagating them through a classifier layer. 
    I do that for each individual target separately.
    '''

    def __init__(self, config):
        super().__init__()

        self.config, self.hyperparameters = config
        self.dropout_rate_bert = self.hyperparameters['bert_dropout']
        self.dropout_rate_hidden = self.hyperparameters['hidden_dropout']


        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.bert.config = self.bert.config.from_dict(self.config)

        self.bertlayer = BertLayer(self.bert.config)
        self.pooling_layer = nn.Sequential(nn.Linear(768, 768, bias=self.hyperparameters['bias']), nn.Tanh())
        self.dropout = nn.Dropout(self.dropout_rate_hidden)
        self.classifier = nn.Linear(768, 4)


    def forward(self, sentences, targets, masks, training=False):
        if training:
            dropout_rate = self.dropout_rate_bert
            self.config['attention_probs_dropout_prob'] = dropout_rate
            self.config['hidden_dropout_prob'] = dropout_rate
            self.bert.config = self.bert.config.from_dict(self.config)
        else:
            dropout_rate = 0.0
            self.config['attention_probs_dropout_prob'] = dropout_rate
            self.config['hidden_dropout_prob'] = dropout_rate
            self.bert.config = self.bert.config.from_dict(self.config)

        outputs = self.bert(sentences, masks)[0]

        absorbing_element = torch.zeros((outputs.size(0), 1, outputs.size(2))).to(outputs.device)
        target_representations = torch.cat([outputs, absorbing_element], dim=1)

        target_representations = target_representations[[[i]*40 for i in range(outputs.size(0))], targets]

        aggregation = torch.mean(target_representations, dim=1)

        pooling = self.pooling_layer(aggregation)

        if training: pooling = self.dropout(pooling)

        logits = self.classifier(pooling)
        outputs = torch.softmax(logits, dim=-1)

        return logits, outputs


class myLitASC(pl.LightningModule):

    def __init__(self, config, label_weights):
        super().__init__()

        self.sentiment_classifier = myASC(config)

        self.hyperparameters = config[1]
        self.hidden_lr = self.hyperparameters['hidden_lr']
        self.bert_lr = self.hyperparameters['bert_lr']
        self.step_size = self.hyperparameters['step_size']
        self.lr_gamma = self.hyperparameters['lr_gamma']

        self.label_weights = torch.tensor(label_weights) if label_weights else None
        self.loss = CrossEntropyLoss(weight=self.label_weights, ignore_index=-1)

    def forward(self, x):
        sentence = x['sentence']
        mask = x['mask']
        target_indices = x['target_indices']

        logits, outputs = self.sentiment_classifier(sentence, target_indices, mask)
        return logits, outputs

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {'hp/dropout_hidden': self.sentiment_classifier.dropout_rate_hidden, 
                                                   'hp/dropout_bert': self.sentiment_classifier.dropout_rate_bert
                                                   })

    def training_step(self, batch, batch_idx):
        sentences = batch['sentence']
        masks = batch['mask']
        target_indices = batch['target_indices']
        labels = batch['aspect_polarities']

        logits, outputs = self.sentiment_classifier(sentences, target_indices, masks, training=True)

        loss = self.loss(logits.view(-1, 4), labels.view(-1))

        predictions = torch.argmax(outputs, dim=-1)

        balanced_accuracy = balanced_accuracy_score(labels.cpu(), predictions.cpu())
        accuracy = accuracy_score(labels.cpu(), predictions.cpu())
        f1 = f1_score(labels.cpu(), predictions.cpu(), average='macro')

        self.log('hp/dropout_hidden', self.sentiment_classifier.dropout_rate_hidden)
        self.log('hp/dropout_bert', self.sentiment_classifier.dropout_rate_bert)

        self.log('train/f1', f1)
        self.log('train/balanced_accuracy', balanced_accuracy)
        self.log('train/accuracy', accuracy)
        self.log('train/loss', loss)

        return {'loss' : loss, 'f1' : f1}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_f1 = torch.tensor([x['f1'] for x in outputs]).mean()
        self.log('train/avg_loss', avg_loss)
        self.log('train/avg_f1', avg_f1)

    def validation_step(self, batch, batch_idx):
        sentences = batch['sentence']
        masks = batch['mask']
        target_indices = batch['target_indices']
        labels = batch['aspect_polarities']

        logits, outputs = self.sentiment_classifier(sentences, target_indices, masks)

        loss = self.loss(logits.view(-1, 4), labels.view(-1))

        predictions = torch.argmax(outputs, dim=-1)

        balanced_accuracy = balanced_accuracy_score(labels.cpu(), predictions.cpu())
        accuracy = accuracy_score(labels.cpu(), predictions.cpu())
        f1 = f1_score(labels.cpu(), predictions.cpu(), average='macro')

        self.log('val/f1', f1)
        self.log('val/balanced_accuracy', balanced_accuracy)
        self.log('val/accuracy', accuracy)
        self.log('val/loss', loss)

        return {'val_loss' : loss, 'val_f1': f1}

    def validation_epoch_end(self, outputs):
        avg_f1 = torch.tensor([x['val_f1'] for x in outputs]).mean()
        self.log('val/epoch_f1', avg_f1)
        log = {'avg_f1' : avg_f1}
        return {'avg_f1' : avg_f1, 'log' : log}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([

                                    {'params' : self.sentiment_classifier.pooling_layer.parameters()},
                                    {'params' : self.sentiment_classifier.classifier.parameters()},
                                    {'params' : self.sentiment_classifier.bert.parameters(), 'lr' : self.bert_lr}
                                     ], lr=self.hidden_lr)

        lr_scheduler = {
            'scheduler' : torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.lr_gamma),
            'name' : 'learning_rate'
        }
        return [optimizer], [lr_scheduler]


class myASC_Multi(ptMeta):
    '''
    My own approach inspired from "IMPROVING BERT PERFORMANCE FOR ASPECT-BASED SENTIMENT ANALYSIS".
    In their approach they extract the sentiment without considering the target word.
    Thus, if there are more than one aspect sentiments in the sentences their model would fail to detect all of them.
    Instead, I use the pooling layer and use it not only on the [CLS] token but on the target tokens as well.
    Afterwards aggregating them resulting representations, and propagating them through a classifier layer. 
    I do that for each individual target separately.

    '''

    def __init__(self, config):
        super().__init__()

        self.config, self.hyperparameters = config
        self.dropout_rate_bert = self.hyperparameters['bert_dropout']
        self.dropout_rate_hidden = self.hyperparameters['hidden_dropout']


        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.bert.config = self.bert.config.from_dict(self.config)

        self.pooling_layer = nn.Sequential(nn.Linear(768, 768, bias=self.hyperparameters['bias']), nn.Tanh())
        self.dropouts = nn.ModuleList()
        self.bert_layers = nn.ModuleList()

        for i in range(4):
            self.bert_layers.append(BertLayer(self.bert.config))
            self.dropouts.append(nn.Dropout(self.dropout_rate_hidden))
        self.classifier = nn.Linear(768, 4)


    def forward(self, sentences, targets, masks, training=False):
        if training:
            dropout_rate = self.dropout_rate_bert
            self.config['attention_probs_dropout_prob'] = dropout_rate
            self.config['hidden_dropout_prob'] = dropout_rate
            self.bert.config = self.bert.config.from_dict(self.config)
        else:
            dropout_rate = 0.0
            self.config['attention_probs_dropout_prob'] = dropout_rate
            self.config['hidden_dropout_prob'] = dropout_rate
            self.bert.config = self.bert.config.from_dict(self.config)

        outputs = self.bert(sentences, masks, output_hidden_states=True)
        outputs = outputs[2][-4:]
        logits_ = []
        for i, bert_output in enumerate(outputs):
            absorbing_element = torch.zeros((bert_output.size(0), 1, bert_output.size(2))).to(bert_output.device)
            target_representations = torch.cat([bert_output, absorbing_element], dim=1)
            target_representations = target_representations[[[i]*40 for i in range(bert_output.size(0))], targets]
            aggregation = torch.mean(target_representations, dim=1, keepdim=True)
            output = self.bert_layers[i](aggregation)[0].squeeze()
            output = self.pooling_layer(output)
            logits = self.classifier(output)
            logits_.append(logits)

        avg_logits = torch.mean(torch.stack(logits_), dim=0)

        avg_outputs = torch.softmax(avg_logits, dim=-1)

        return avg_logits, avg_outputs, logits_


class myLitASC_Multi(pl.LightningModule):

    def __init__(self, config, label_weights):
        super().__init__()

        self.sentiment_classifier = myASC_Multi(config)

        self.hyperparameters = config[1]
        self.hidden_lr = self.hyperparameters['hidden_lr']
        self.bert_lr = self.hyperparameters['bert_lr']
        self.step_size = self.hyperparameters['step_size']
        self.lr_gamma = self.hyperparameters['lr_gamma']

        self.label_weights = torch.tensor(label_weights) if label_weights else None
        self.losses = nn.ModuleList()
        for i in range(4):
            self.losses.append(CrossEntropyLoss(weight=self.label_weights, ignore_index=-1))

    def forward(self, x):
        sentence = x['sentence']
        mask = x['mask']
        target_indices = x['target_indices']

        avg_logits, avg_outputs, logits_list = self.sentiment_classifier(sentence, target_indices, mask)

        return avg_logits, avg_outputs, logits_list

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {'hp/dropout_hidden': self.sentiment_classifier.dropout_rate_hidden, 
                                                   'hp/dropout_bert': self.sentiment_classifier.dropout_rate_bert
                                                   })

    def training_step(self, batch, batch_idx):
        sentences = batch['sentence']
        masks = batch['mask']
        target_indices = batch['target_indices']
        labels = batch['aspect_polarities']

        avg_logits, avg_outputs, logits_list = self.sentiment_classifier(sentences, target_indices, masks, training=True)

        losses = []
        for i in range(4):
            losses.append(self.losses[i](logits_list[i], labels))
        loss = torch.mean(torch.stack(losses), dim=0)
        predictions = torch.argmax(avg_outputs, dim=-1)

        balanced_accuracy = balanced_accuracy_score(labels.cpu(), predictions.cpu())
        accuracy = accuracy_score(labels.cpu(), predictions.cpu())
        f1 = f1_score(labels.cpu(), predictions.cpu(), average='macro')

        self.log('hp/dropout_hidden', self.sentiment_classifier.dropout_rate_hidden)
        self.log('hp/dropout_bert', self.sentiment_classifier.dropout_rate_bert)

        self.log('train/f1', f1)
        self.log('train/balanced_accuracy', balanced_accuracy)
        self.log('train/accuracy', accuracy)
        self.log('train/loss', loss)

        return {'loss' : loss, 'f1' : f1}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_f1 = torch.tensor([x['f1'] for x in outputs]).mean()
        self.log('train/avg_loss', avg_loss)
        self.log('train/avg_f1', avg_f1)

    def validation_step(self, batch, batch_idx):
        sentences = batch['sentence']
        masks = batch['mask']
        target_indices = batch['target_indices']
        labels = batch['aspect_polarities']

        avg_logits, avg_outputs, logits_list = self.sentiment_classifier(sentences, target_indices, masks)

        losses = []
        for i in range(4):
            losses.append(self.losses[i](logits_list[i], labels))
        loss = torch.mean(torch.stack(losses), dim=0)
        predictions = torch.argmax(avg_outputs, dim=-1)
        balanced_accuracy = balanced_accuracy_score(labels.cpu(), predictions.cpu())
        accuracy = accuracy_score(labels.cpu(), predictions.cpu())
        f1 = f1_score(labels.cpu(), predictions.cpu(), average='macro')

        self.log('val/f1', f1)
        self.log('val/balanced_accuracy', balanced_accuracy)
        self.log('val/accuracy', accuracy)
        self.log('val/loss', loss)

        return {'val_loss' : loss, 'val_f1': f1}

    def validation_epoch_end(self, outputs):
        avg_f1 = torch.tensor([x['val_f1'] for x in outputs]).mean()
        self.log('val/epoch_f1', avg_f1)
        log = {'avg_f1' : avg_f1}
        return {'avg_f1' : avg_f1, 'log' : log}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
                                    {'params' : self.sentiment_classifier.bert_layers.parameters()},
                                    {'params' : self.sentiment_classifier.pooling_layer.parameters()},
                                    {'params' : self.sentiment_classifier.classifier.parameters()},
                                    {'params' : self.sentiment_classifier.bert.parameters(), 'lr' : self.bert_lr}
                                     ], lr=self.hidden_lr)

        lr_scheduler = {
            'scheduler' : torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.lr_gamma),
            'name' : 'learning_rate'
        }
        return [optimizer], [lr_scheduler]