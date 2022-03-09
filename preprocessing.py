import numpy as np
import json, random, os, sys, string

from transformers import BertTokenizer
os.chdir(os.path.dirname(os.path.realpath(__file__)))

DOMAINS = ['laptops', 'restaurants']
DATASETS = ['train', 'dev']

def stats():

    ### EXTRACT
    ## Number of sentences
    # Lengths of sentences
    # Number of targets per sentence
    # Lengths of targets
    ## Numbers of target tags

    domain = DOMAINS[0]
    dataset = DATASETS[0]

    for domain in DOMAINS:
        domain_n_sentences = 0
        domain_n_targets = 0
        for dataset in DATASETS:
            filename = '{}_{}.json'.format(domain, dataset)
            with open('data/' + filename, 'r') as reader:
                raw_data = json.load(reader)
                reader.close()

            has_categories = False
            category_types, category_polarities = {}, {}
            n_sentences = len(raw_data)
            domain_n_sentences += n_sentences
            sentence_lengths, n_targets_per_sentence, target_lengths = [], [], []
            n_target_tags = {'neutral' : 0, 'positive' : 0, 'negative' : 0}
            targetless = 0

            for raw in raw_data:
                targets = raw['targets']
                sentence = raw['text']
                sentence_lengths.append(len(sentence.split()))
                n_targets_per_sentence.append(len(targets))

                try:
                    categories = raw['categories']

                    for category in categories:
                        try:
                            category_types[category[0]] += 1
                        except:
                            category_types[category[0]] = 1

                        try:
                            category_polarities[category[1]] += 1
                        except:
                            category_polarities[category[1]] = 1
                    has_categories = True
                except KeyError:
                    pass

                if len(targets) > 0:
                    for target in targets:
                        target_lengths.append(len(target[1].split()))
                        try:
                            n_target_tags[target[2]] += 1
                        except KeyError:
                            n_target_tags[target[2]] = 1
                else:
                    targetless += 1
            else:
                print(filename)
            domain_n_targets += np.sum(n_targets_per_sentence)
            print('Number of sentences: ', n_sentences)
            print('Number of sentences without targets: ', targetless)
            print('Total number of targets: ', np.sum(n_targets_per_sentence))
            print('Number of target tags:')
            print(n_target_tags)
            if has_categories:
                print('Number of categories: ', len(list(category_types.keys())))
                print('Categories: ')
                print(category_types)
                print('Category polarities: ')
                print(category_polarities)
            print('Sentence lengths max/min/mean/median: ')
            print(np.max(sentence_lengths), np.min(sentence_lengths), np.mean(sentence_lengths), np.median(sentence_lengths))
            print('Number of targets per sentence max/min/mean/median:')
            print(np.max(n_targets_per_sentence), np.min(n_targets_per_sentence), np.mean(n_targets_per_sentence), np.median(n_targets_per_sentence))
            print('Target lengths max/min/mean/median:')
            print(np.max(target_lengths), np.min(target_lengths), np.mean(target_lengths), np.median(target_lengths))
            print('*'*50)

        print(f'DOMAIN {domain}')
        print('Number of sentences: ', domain_n_sentences)
        print('Total number of targets: ', domain_n_targets)
        print('-'*50)

def bert_preprocessing(domains, datasets):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=False)
    polarity_tags = ['neutral', 'negative', 'positive', 'conflict']
    category_tags = ['anecdotes/miscellaneous', 'food', 'service', 'price', 'ambience']

    for domain in domains:
        has_categories = True if domain == 'restaurants' else False

        for dataset in datasets:
            print(f'DOMAIN {domain} DATASET {dataset}')
            filename = f'data/{domain}_{dataset}.json'
            finitoset = {}
            sentences, ae_labels, aspect_polarities, category_targets, category_polarities = [], [], [], [], []
            target_indices = []
            sentence_lengths = []
            c,t = 0, 0
            with open(filename, 'r') as reader:
                raw_data = json.load(reader)
                reader.close()
            control = None
            for i, raw in enumerate(raw_data):
                text = raw['text']
                targets = raw['targets']

                if has_categories:
                    categories = raw['categories']
                    c_targets, c_polarities = [], []
                    for category in categories:
                        c_targets.append(category_tags.index(category[0]))
                        c_polarities.append(polarity_tags.index(category[1]))
                    category_targets.append(c_targets)
                    category_polarities.append(c_polarities)
                else:
                    category_targets.append([])
                    category_polarities.append([])
                # print(text)
                # print(targets)
                all_indices = []
                target_bios = []
                aspect_polarity = []

                # Sort targets according to the indices
                if len(targets) > 1:
                    targets = sorted(targets, key=lambda x: x[0][0], reverse=False)

                text1 = text
                # print('\n')
                t_text = tokenizer.encode(text, add_special_tokens=False)
                for target in targets:
                    t+= 1
                    indices = target[0]
                    # all_indices.extend(indices)
                    aspect = text[indices[0]:indices[1]]
                    assert aspect == target[1]
                    sentiment = target[2]
                    # print(aspect)
                    tokenized_aspect = tokenizer.encode(aspect, add_special_tokens=False)

                    pre_target = text[:indices[0]]
                    # print(pre_target)
                    pre_target = tokenizer.encode(pre_target, add_special_tokens=False)
                    # print(len(pre_target))
                    pre_w_target = pre_target + tokenized_aspect
                    if not pre_w_target == t_text[:len(pre_w_target)]:
                        # These are targets that cannot be found in a sentence after tokenization
                        # Since there are not many of them, they will be discarded for the training
                        c+=1 
                        continue

                    new_indices = [len(pre_target) + i + 1 for i in range(len(tokenized_aspect))]


                    # BIO Tag 
                    target_bio = [1] * len(tokenized_aspect)
                    target_bio[0] = 2
                    target_bios.append(target_bio)

                    aspect_polarity.append(polarity_tags.index(sentiment))

                    all_indices.append(new_indices)
                    fool = tokenizer.decode(tokenized_aspect)
                    text1 = text[:indices[0]] + aspect + text[indices[1]:]
                    tokenized_sentence = tokenizer.encode(text1, add_special_tokens=True)

                    assert tokenized_sentence[new_indices[0]] == tokenized_aspect[0], f'${tokenized_sentence}$ - ${tokenized_aspect}$'
                    assert ' '.join(aspect.split()) == fool, f'${aspect}$ - ${fool}$'
                
                tokenized_sentence = tokenizer.encode(text, add_special_tokens=True)
                target_indices.append(all_indices)
                
                # BIO Tags
                bio_tags = [0] * len(tokenized_sentence)
                assert len(all_indices) == len(target_bios) == len(aspect_polarity)
                for indices, bios in zip(all_indices, target_bios):
                    for index, bio in zip(indices, bios):
                        bio_tags[index] = bio
                sentences.append(tokenized_sentence)
                sentence_lengths.append(len(tokenized_sentence))
                aspect_polarities.append(aspect_polarity)
                ae_labels.append(bio_tags)

            print(f'{c}/{t}')
            finitoset['sentences'] = sentences
            finitoset['ae_labels'] = ae_labels
            finitoset['target_indices'] = target_indices
            finitoset['aspect_polarities'] = aspect_polarities
            finitoset['sentence_lengths'] = sentence_lengths
            finitoset['category_targets'] = category_targets
            finitoset['category_polarities'] = category_polarities

            with open('data/_bert_preprocessed_{}_{}.json'.format(domain, dataset), 'w') as writer:
                json.dump(finitoset, writer)
                writer.close()
            print('Max sentence_lengths:  ', np.max(sentence_lengths))     # 96
            print('Min sentence_lengths:  ', np.min(sentence_lengths))     # 4
            print('Mean sentence_lengths: ', np.mean(sentence_lengths))    # 20.222
            print('Med sentence_lengths:  ', np.median(sentence_lengths))  # 18

def preprocessing(domains, datasets):
    polarity_tags = ['neutral', 'negative', 'positive', 'conflict']
    category_tags = ['anecdotes/miscellaneous', 'food', 'service', 'price', 'ambience']
    fake_aspect = 'xxxtheteapotreturnsxxx'
    mapping = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    for domain in domains:
        has_categories = True if domain == 'restaurants' else False

        for dataset in datasets:
            print(f'DOMAIN {domain} DATASET {dataset}')
            filename = f'data/{domain}_{dataset}.json'
            finitoset = {}
            sentences, ae_labels, aspect_polarities, category_targets, category_polarities = [], [], [], [], []
            sentence_lengths = []
            with open(filename, 'r') as reader:
                raw_data = json.load(reader)
                reader.close()
            control = None
            for i, raw in enumerate(raw_data):
                text = raw['text']
                targets = raw['targets']

                if has_categories:
                    categories = raw['categories']

                sentence1 = text.translate(mapping)
                sentence1 = sentence1.split()
                sentence1 = [word for word in sentence1 if word]
                sentence_lengths.append(len(sentence1))
                sentences.append(sentence1)
                control = sentence1
                bio_tags = [0] * len(sentence1)
                sentence = text
                aspect_polarity = []
                if len(targets):
                    for target in targets:
                        indices = target[0]
                        aspect = target[1].split()
                        sentiment = target[2]
                        aspect_polarity.append(polarity_tags.index(sentiment))

                        fake_sentence = sentence
                        fake_sentence = fake_sentence[:indices[0]] + ' ' + fake_aspect + ' ' + fake_sentence[indices[1]:]
                        fake_sentence = fake_sentence.translate(mapping)
                        fake_sentence = fake_sentence.split()
                        fake_sentence = [word for word in fake_sentence if word]
                        aspect_index = fake_sentence.index(fake_aspect)
                        fake_sentence = fake_sentence[:aspect_index] + aspect + fake_sentence[aspect_index+1:]

                        
                        # Aspect Extraction objects
                        # BIO tags -> O 0, I 1, B 2
                        try:
                            for i, j in enumerate(aspect):
                                if not i: 
                                    bio_tags[aspect_index] = 2
                                else:
                                    bio_tags[aspect_index+i] = 1
                        except IndexError:
                            print(raw)
                            print(sentence)
                            print(fake_sentence, len(fake_sentence))
                            print(bio_tags, len(bio_tags))
                            print(aspect)
                            print(aspect_index)
                            print(control)
                            exit()

                ae_labels.append(bio_tags)
                aspect_polarities.append(aspect_polarity)

            finitoset['sentences'] = sentences
            finitoset['ae_labels'] = ae_labels
            finitoset['aspect_polarities'] = aspect_polarities
            finitoset['sentence_lengths'] = sentence_lengths

            with open('data/preprocessed_{}_{}.json'.format(domain, dataset), 'w') as writer:
                json.dump(finitoset, writer)
                writer.close()


def asc_dataset(domains, datasets):
    for domain in domains:
        has_categories = True if domain == 'restaurants' else False

        for dataset in datasets:
            print(f'DOMAIN {domain} DATASET {dataset}')
            filename = f'data/_bert_preprocessed_{domain}_{dataset}.json'
            with open(filename, 'r') as reader:
                data = json.load(reader)
                reader.close()

            sentences = data['sentences']
            ae_labels = data['ae_labels']
            target_indices = data['target_indices']
            aspect_polarities = data['aspect_polarities']
            sentence_lengths = data['sentence_lengths']
            category_targets = data['category_targets']
            category_polarities = data['category_polarities']
            del data
            assert len(target_indices) == len(aspect_polarities)
            new_sentences, new_ae_labels, new_aspect_polarities, new_category_targets, new_category_polarities = [], [], [], [], []
            new_target_indices = []
            new_sentence_lengths = []
            for i, (sentence, ae_label, target_index, aspect_polarity, sentence_length, category_target, category_polarity) in enumerate(zip(sentences, ae_labels, target_indices, aspect_polarities, sentence_lengths, category_targets, category_polarities)):
                for index, polarity in zip(target_index, aspect_polarity):
                    # This loop automatically discards sentences WITHOUT A TARGET
                    new_sentences.append(sentence)
                    new_ae_labels.append(ae_label)
                    new_target_indices.append(index)
                    new_aspect_polarities.append(polarity)
                    new_sentence_lengths.append(sentence_length)
                    new_category_targets.append(category_target)
                    new_category_polarities.append(category_polarity)

            data = {
                'sentences' : new_sentences,
                'ae_labels' : new_ae_labels,
                'target_indices' : new_target_indices,
                'aspect_polarities' : new_aspect_polarities,
                'sentence_lengths' : new_sentence_lengths,
                'category_targets' : new_category_targets,
                'category_polarities' : new_category_polarities,
            }
            print('Number of previous data points: ', len(sentences))
            print('Number of new data points : ', len(new_sentences))
            filename = f'data/_bert_asc_preprocessed_{domain}_{dataset}.json'
            with open(filename, 'w') as writer:
                json.dump(data, writer)
                writer.close()
            del data

def merge_datasets(domains, datasets):
    TASKS = ['ae', 'asc']
    for task in TASKS:
        for dataset in datasets:
            merged_data = {
                'sentences' : [],
                'ae_labels' : [],
                'target_indices' : [],
                'aspect_polarities' : [],
                'sentence_lengths' : []
            }
            for domain in domains:
                print(f'DOMAIN {domain} DATASET {dataset}')
                if task == 'asc':
                    filename = f'data/_bert_asc_preprocessed_{domain}_{dataset}.json'
                elif task == 'ae':
                    filename = f'data/_bert_preprocessed_{domain}_{dataset}.json'
                    
                with open(filename, 'r') as reader:
                    data = json.load(reader)
                    reader.close()
                merged_data['sentences'].extend(data['sentences'])
                merged_data['ae_labels'].extend(data['ae_labels'])
                merged_data['target_indices'].extend(data['target_indices'])
                merged_data['aspect_polarities'].extend(data['aspect_polarities'])
                merged_data['sentence_lengths'].extend(data['sentence_lengths'])
            print('Merged data number of data: ', len(merged_data['sentences']))
            print(f'DOMAIN {domain} DATASET {dataset} TASK {task}')
            filename = f'data/merged_{task}_{dataset}.json'
            with open(filename, 'w') as writer:
                json.dump(merged_data, writer)
                writer.close()
            print(f'Saved {filename}\n')
if __name__ == '__main__':
    # stats()
    # preprocessing(DOMAINS, DATASETS)
    # bert_preprocessing(DOMAINS, DATASETS)
    # asc_dataset(DOMAINS, DATASETS)
    merge_datasets(DOMAINS, DATASETS)