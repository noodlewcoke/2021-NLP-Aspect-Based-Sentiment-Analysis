# 2021-NLP-Aspect-Based-Sentiment-Analysis

2021

The implementation of Aspect Based Sentiment Analysis (ABSA) framework for an NLP course.

The project was implemented and optimized with Pytorch Lightning.

The ABSA framework consists of two models: Aspect Extraction (AE) and Aspect Sentiment Analysis (ASC). Given a sentence the AE model is used to determine which word or word groups in the sentence have a sentimental association. Then, the ASC model is used to identify said sentiment.

Both the AE and the ASC models in the project employ BERT [(Devlin et al., 2018)](https://arxiv.org/abs/1810.04805) model to obtain contextual word embeddings, therefore the text data and the target word data need to be processed accordingly. In that order, the text data and the target data (for the ASC module) was processed using BertTokenizer module with the ’bert-base-cased’ configuration, since the targets may contain upper cased words and this information can be exploited. Then, the indices in the tokenized sentence for the tokenized words were extracted. These indices information will be used the optimization of both tasks. Further, the different steps taken for preprocessing for the two tasks are described.

The models for both tasks were built by following the approaches developed by [(Karimi et al., 2020)](https://arxiv.org/abs/2010.11731).

