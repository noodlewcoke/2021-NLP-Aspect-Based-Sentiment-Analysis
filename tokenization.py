import numpy as np
import json, os, sys, random
import torch
from transformers import  BertTokenizer

os.chdir(os.path.dirname(os.path.realpath(__file__)))

DOMAINS = ['laptops', 'restaurants']
DATASETS = ['train', 'dev']

def tokenize(domains, datasets, special_tokens=False):
    '''
    Sentence tokenization and padding
    '''

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    max_len = 90
    for domain in domains:

        for dataset in datasets:

            filename = f'preprocessed_{domain}_{dataset}.json'
            print('Processing ', filename)
            with open('data/'+filename, 'r') as reader:
                preprocessed_data = json.load(reader)
                reader.close()
            
            sentences = preprocessed_data['sentences']
            ae_labels = preprocessed_data['ae_labels']
            aspect_polarities = preprocessed_data['aspect_polarities']
            sentence_lengths = preprocessed_data['sentence_lengths']
            target_indices = preprocessed_data['target_indices']

            del preprocessed_data

            masks = []
            tokenized_sentences = []
            padded_labels = []
            extra_labels = []
            for sentence, ae_label in zip(sentences, ae_labels):
                t_sentence = tokenizer.convert_tokens_to_ids(sentence)
                assert len(t_sentence) == len(sentence)
                if special_tokens:
                    t_sentence = [101] + t_sentence + [102]
                    ae_label = [-1] + ae_label + [-1]

                # sentence padding and mask
                pad = [0] * (max_len - len(t_sentence))
                mask = [1] * len(t_sentence) + pad
                t_sentence += pad

                assert len(t_sentence) == len(mask)
                tokenized_sentences.append(t_sentence)
                masks.append(mask)

                # label padding
                pad = [-1] * (max_len - len(ae_label))
                p_label = ae_label + pad
                padded_labels.append(p_label)

                if special_tokens:
                    ae_label[0] = 0
                    ae_label[-1] = 0 
                pad = [0] * (max_len - len(ae_label))
                p_label = ae_label + pad
                extra_labels.append(p_label)

            data = {
                'sentences' : tokenized_sentences,
                'ae_labels' : padded_labels,
                'target_indices' : target_indices,
                'extra_labels' : extra_labels,
                'aspect_polarities' : aspect_polarities,
                'sentence_lengths' : sentence_lengths,
                'masks' : masks
            }
            with open('data/tokenized_{}_{}.json'.format(domain, dataset), 'w') as writer:
                json.dump(data, writer)
                writer.close()


def bert_pad(domains, datasets):
    '''
    Sentence tokenization and padding
    '''

    max_len = 100
    TASKS = ['ae' , 'asc']
    # for domain in domains:
    for task in TASKS:

        for dataset in datasets:

            # filename = f'_bert_preprocessed_{domain}_{dataset}.json'
            filename = f'merged_{task}_{dataset}.json'

            print('Processing ', filename)
            with open('data/'+filename, 'r') as reader:
                preprocessed_data = json.load(reader)
                reader.close()
            
            sentences = preprocessed_data['sentences']
            ae_labels = preprocessed_data['ae_labels']
            aspect_polarities = preprocessed_data['aspect_polarities']
            sentence_lengths = preprocessed_data['sentence_lengths']
            target_indices = preprocessed_data['target_indices']
            del preprocessed_data
            masks = []
            tokenized_sentences = []
            padded_labels = []
            extra_labels = []
            padded_targets = []
            for sentence, ae_label, targets in zip(sentences, ae_labels, target_indices):
                # sentence padding and mask
                pad = [0] * (max_len - len(sentence))
                mask = [1] * len(sentence) + pad
                sentence += pad

                assert len(sentence) == len(mask)
                tokenized_sentences.append(sentence)
                masks.append(mask)

                # label padding
                pad = [-1] * (max_len - len(ae_label))
                p_label = ae_label + pad
                padded_labels.append(p_label)

                pad = [0] * (max_len - len(ae_label))
                p_label = ae_label + pad
                extra_labels.append(p_label)

                # target index padding 
                assert len(targets) < 40
                padded_target = targets + [-1] * (40 - len(targets))
                padded_targets.append(padded_target)
            data = {
                'sentences' : tokenized_sentences,
                'ae_labels' : padded_labels,
                'target_indices' : padded_targets,
                'extra_labels' : extra_labels,
                'aspect_polarities' : aspect_polarities,
                'sentence_lengths' : sentence_lengths,
                'masks' : masks
            }
            # with open('data/_bert_tokenized_{}_{}.json'.format(domain, dataset), 'w') as writer:
            with open('data/tokenized_merged_{}_{}.json'.format(task, dataset), 'w') as writer:

                json.dump(data, writer)
                writer.close()
  

if __name__ == '__main__':
    # tokenize(DOMAINS, DATASETS, special_tokens=True)
    bert_pad(DOMAINS, DATASETS)


