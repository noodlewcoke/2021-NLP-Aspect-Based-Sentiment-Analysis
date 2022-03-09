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
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
os.chdir(os.path.dirname(os.path.realpath(__file__)))

class ptMeta(nn.Module):

    def __init__(self):
        super(ptMeta, self).__init__()

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location='cpu'))


# ASPECT EXTRACTION

class simpleAE(ptMeta):

    def __init__(self,
                config,
                lr = 1e-3,
                bias=True):
        super(simpleAE, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.config = config
        self.bert.config = self.bert.config.from_dict(self.config)
        self.classifier = nn.Sequential(
                                        nn.Linear(768, 768, bias=bias),
                                        nn.ReLU(),
                                        nn.Linear(768, 768, bias=bias),
                                        nn.ReLU(),
                                        nn.Linear(768, 768, bias=bias),
                                        nn.ReLU(),
                                        nn.Linear(768, 3, bias=bias)
                                        )
        
        self.loss = CrossEntropyLoss(ignore_index=3)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)

    def forward(self, sentences, masks):
        x = self.bert(sentences, masks)
        x = x[0]
        logits = self.classifier(x)
        output = torch.softmax(logits, dim=-1)

        return logits, output

    def update(self, sentences, masks, labels):
        logits, _ = self(sentences, masks)

        self.optimizer.zero_grad()
        loss = self.loss(logits.view(-1, 3), labels.view(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()


class betterAE(ptMeta):

    def __init__(self, config, mode='p'):
        super(betterAE, self).__init__()

        self.mode = mode
        self.config, self.hyperparameters = config
        self.dropout_rate_bert = self.hyperparameters['bert_dropout']
        self.dropout_rate_hidden = self.hyperparameters['hidden_dropout']

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.bert.config = self.bert.config.from_dict(self.config)

        self.hidden_bert1 = BertLayer(self.bert.config)
        self.hidden_bert2 = BertLayer(self.bert.config)
        self.hidden_bert3 = BertLayer(self.bert.config)
        self.output_bert = BertLayer(self.bert.config)

        self.dropout2 = nn.Dropout(self.dropout_rate_hidden)
        self.dropout3 = nn.Dropout(self.dropout_rate_hidden)
        self.dropout4 = nn.Dropout(self.dropout_rate_hidden)
        self.dropout1 = nn.Dropout(self.dropout_rate_hidden)

        self.linear = nn.Linear(768, 3, bias=self.hyperparameters['bias'])

        self.crf = nn.ModuleList()
        for i in range(4):
            self.crf.append(CRF(3))


    def forward(self, sentences, masks, training=False):
        if self.mode == 'p':
            logits, pre_logits = self.parallel_aggregation(sentences, masks, training)
            output = torch.softmax(logits, dim=-1)
            return logits, output, pre_logits
        elif self.mode == 'h':
            logits, pre_logits = self.hierarchical_aggregation(sentences, masks)
            output = torch.softmax(logits, dim=-1)
            return logits, output, pre_logits

    def parallel_aggregation(self, sentences, masks, training=False):
        x = self.bert(sentences, masks, output_hidden_states=True, output_attentions=True)

        hidden_state1, hidden_state2, hidden_state3, output_state = x[2][-4:]

        x1 = self.hidden_bert1(hidden_state1)[0]
        x2 = self.hidden_bert2(hidden_state2)[0]
        x3 = self.hidden_bert3(hidden_state3)[0]
        x4 = self.output_bert(output_state)[0]
        if training:
            x1 = self.dropout1(x1)
            x2 = self.dropout2(x2)
            x3 = self.dropout3(x3)
            x4 = self.dropout4(x4)
        logits1 = self.linear(x1)
        logits2 = self.linear(x2)
        logits3 = self.linear(x3)
        logits4 = self.linear(x4)
        pre_logits = [logits1, logits2, logits3, logits4]

        logits = torch.sum(torch.stack(pre_logits), dim=0)/4

        return logits, pre_logits

    def hierarchical_aggregation(self, sentences, masks):
        x = self.bert(sentences, masks, output_hidden_states=True)
        hidden_state1, hidden_state2, hidden_state3, output_state = x[2][-4:]

        output4 = self.output_bert(output_state)[0]
        logits4 = self.linear(output4)

        output4 = output4.clone() + hidden_state3
        output3 = self.hidden_bert3(output4)[0]
        logits3 = self.linear(output3)

        output3 = output3.clone() + hidden_state2
        output2 = self.hidden_bert2(output3)[0]
        logits2 = self.linear(output2)

        output2 = output2.clone() + hidden_state1
        output1 = self.hidden_bert1(output2)[0]
        logits1 = self.linear(output1)

        pre_logits = [logits1, logits2, logits3, logits4]
        logits = torch.sum(torch.stack(pre_logits), dim=0)/4

        return logits, pre_logits


class LitAspectExtractorSimple(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.extractor = simpleAE(config)
        self.loss = CrossEntropyLoss(ignore_index=-1)

    def forward(self, x):
        sentence = x['sentence']
        mask = x['mask']
        
        logits, outputs = self.extractor(sentence, mask)
        
        return logits, outputs

    def training_step(self, batch, batch_idx):
        sentences = batch['sentence']
        masks = batch['mask']
        labels = batch['label']

        logits, outputs = self.extractor(sentences, masks)

        loss = self.loss(logits.view(-1,3), labels.view(-1))
        predictions = torch.argmax(outputs, dim=-1)
        c_prediction, c_label = cleanse_predictions(predictions, labels, -1)
        accuracy = balanced_accuracy_score(c_label, c_prediction)

        precision, recall, f1 = index_based_accuracy(sentences, predictions, labels)
        self.log('train/precision', precision)
        self.log('train/recall', recall)
        self.log('train/f1', f1)
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
        labels = batch['label']

        logits, outputs = self.extractor(sentences, masks)

        loss = self.loss(logits.view(-1,3), labels.view(-1))
        predictions = torch.argmax(outputs, dim=-1)
        c_prediction, c_label = cleanse_predictions(predictions, labels, -1)
        accuracy = balanced_accuracy_score(c_label, c_prediction)

        precision, recall, f1 = index_based_accuracy(sentences, predictions, labels)
        self.log('val/precision', precision)
        self.log('val/recall', recall)
        self.log('val/f1', f1)
        self.log('val/accuracy', accuracy)
        self.log('val/loss', loss)
        return {'val_loss' : loss, 'val_f1' : f1}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_f1 = torch.tensor([x['val_f1'] for x in outputs]).float().mean()
        self.log('val/epoch_f1' , avg_f1)
        self.log('val/avg_loss' , avg_loss)
        log = {'avg_f1' : avg_f1, 'avg_loss' : avg_loss}
        return {'val_loss' : avg_loss, 'val_f1' : avg_f1, 'log' : log}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.extractor.classifier.parameters(), lr=1e-3)
        return optimizer


class LitAspectExtractorBetter(pl.LightningModule):

    def __init__(self, config, tag_weights, mode='p'):
        super().__init__()
        self.extractor = betterAE(config, mode=mode)

        self.hyperparameters = config[1]
        self.hidden_lr = self.hyperparameters['hidden_lr']
        self.bert_lr = self.hyperparameters['bert_lr']
        self.step_size = self.hyperparameters['step_size']
        self.lr_gamma = self.hyperparameters['lr_gamma']

        self.tag_weights = torch.tensor(tag_weights) if tag_weights else None
        self.loss = CrossEntropyLoss(weight=self.tag_weights, ignore_index=-1)

    def forward(self, x):
        sentence = x['sentence']
        mask = x['mask']
        logits, outputs, pre_logits = self.extractor(sentence, mask)
        return logits, outputs, pre_logits

    def training_step(self, batch, batch_idx):
        sentences = batch['sentence']
        masks = batch['mask']
        labels = batch['label']
        crf_labels = batch['crf_label']
        logits, outputs, pre_logits = self.extractor(sentences, masks, training=True)

        losses = []

        for i in range(len(pre_logits)):
            losses.append(self.extractor.crf[i].forward(pre_logits[i], crf_labels, masks.byte()))
        loss = -torch.mean(torch.stack(losses))

        class_loss = self.loss(logits.view(-1,3), labels.view(-1))
        predictions = torch.argmax(outputs, dim=-1)
        c_prediction, c_label = cleanse_predictions(predictions, labels, -1)
        accuracy = balanced_accuracy_score(c_label, c_prediction)

        precision, recall, f1 = index_based_accuracy(sentences, predictions, labels)
        self.log('train/accuracy', accuracy)
        self.log('train/f1', f1)
        self.log('train/precision', precision)
        self.log('train/recall', recall)
        self.log('train/loss', loss)
        self.log('train/class_loss', class_loss)
        return {'loss' : loss, 'f1' : f1}
        
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_f1 = torch.tensor([x['f1'] for x in outputs]).mean()
        self.log('train/epoch_f1', avg_f1)
        self.log('train/epoch_loss', avg_loss)
    
    def validation_step(self, batch, batch_idx):
        sentences = batch['sentence']
        masks = batch['mask']
        labels = batch['label']
        crf_labels = batch['crf_label']

        logits, outputs, pre_logits = self.extractor(sentences, masks)

        losses = []
        for i in range(len(pre_logits)):
            losses.append(self.extractor.crf[i].forward(pre_logits[i], crf_labels, masks.byte()))
        loss = -torch.mean(torch.stack(losses))
        
        class_loss = self.loss(logits.view(-1,3), labels.view(-1))
        predictions = torch.argmax(outputs, dim=-1)
        c_prediction, c_label = cleanse_predictions(predictions, labels, -1)
        accuracy = balanced_accuracy_score(c_label, c_prediction)
        
        precision, recall, f1 = index_based_accuracy(sentences, predictions, labels)
        self.log('val/accuracy', accuracy)
        self.log('val/f1', f1)
        self.log('val/precision', precision)
        self.log('val/recall', recall)
        self.log('val/loss', loss)
        self.log('val/class_loss', class_loss)
        return {'val_loss' : loss, 'val_f1' : f1}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_f1 = torch.tensor([x['val_f1'] for x in outputs]).float().mean()
        self.log('val/epoch_f1' , avg_f1)
        self.log('val/epoch_loss' , avg_loss)


        log = {'avg_f1' : avg_f1, 'avg_loss' : avg_loss}
        return {'avg_f1' : avg_f1, 'avg_loss' : avg_loss, 'log' : log}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
                                    {'params' : self.extractor.hidden_bert1.parameters() },
                                    {'params' : self.extractor.hidden_bert2.parameters()},
                                    {'params' : self.extractor.hidden_bert3.parameters()},
                                    {'params' : self.extractor.output_bert.parameters()},
                                    {'params' : self.extractor.linear.parameters()},
                                    {'params' : self.extractor.crf.parameters()},
                                    {'params' : self.extractor.bert.parameters(), 'lr' : self.bert_lr}
                                     ], lr=self.hidden_lr)
        return optimizer


