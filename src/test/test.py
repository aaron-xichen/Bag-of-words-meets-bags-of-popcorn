#!/usr/bin/env python
# encoding: utf-8
import pandas as pd
import os
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data


# return word list of a sencence
def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review).getText()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)


# return a list of lists, each list represents a word list in each sentence
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences= []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

# load data
def load_data(path):
    return pd.read_csv(path, header=0, delimiter="\t", quoting=3, encoding='utf-8')



# load dataset
labeledTrainDataPath = os.path.expanduser("../../dataset/labeledTrainData.tsv")
testDataPath = os.path.expanduser("../../dataset/testData.tsv")
unlabeledTrainDataPath = os.path.expanduser("../../dataset/unlabeledTrainData.tsv")
sampleSubmissionPath = os.path.expanduser("../../dataset/sampleSubmission.csv")

train = load_data(labeledTrainDataPath)
test = load_data(testDataPath)

print "labelData:{} , testData: {}".format(train["review"].size, test["review"].size)

sentences = []
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

for review in test["review"]:
    sentences += review_to_sentences(review, tokenizer)
print len(sentences)
