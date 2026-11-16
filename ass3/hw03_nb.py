import math
from collections import Counter
from datasets import load_dataset
from nltk.corpus import stopwords
import string
import re

dataset = load_dataset("imdb")

train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()

################# I have got help from AI in preprocess text functions #################
############################ Comments in logistic file #################################
def remove_extra_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_punctuation(text):
    return remove_extra_whitespace(''.join(' ' if char in string.punctuation else char for char in text))


def remove_numbers(text):
    return remove_extra_whitespace(''.join(' ' if char.isdigit() else char for char in text))


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [' ' if word in stop_words else word for word in words]
    return remove_extra_whitespace(' '.join(filtered_words))

########################################################################################
def preprocess_text(text):
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = text.lower()
    text = remove_stopwords(text)
    text = remove_extra_whitespace(text)

    return text


class NaiveBayesClassifier:
    def __init__(self):
        self.train_df = None
        self.total_pos_words = 0
        self.total_neg_words = 0
        self.vocab_size = 0
        self.vocab = {}
        self.prior_pos = 0
        self.prior_neg = 0
        self.pos_counter = Counter()
        self.neg_counter = Counter()

    def fit(self, train):
        train['text'] = train['text'].apply(preprocess_text)
        self.train_df = train

        pos_texts = self.train_df[self.train_df['label'] == 1]['text']
        neg_texts = self.train_df[self.train_df['label'] == 0]['text']

        self.prior_pos = len(pos_texts) / len(self.train_df)
        self.prior_neg = len(neg_texts) / len(self.train_df)

        for text in pos_texts:
            words = text.split()
            self.total_pos_words += len(words)
            self.pos_counter.update(words)

        for text in neg_texts:
            words = text.split()
            self.total_neg_words += len(words)
            self.neg_counter.update(words)

        for text in train_df['text']:
            words = text.split()
            for word in words:
                self.vocab[word] = self.vocab.get(word, 0) + 1

        self.vocab_size = len(self.vocab) + 1 # + 1 is coming from space because I take .split I loose space char

    def predict(self, text):
        positiveProb = math.log(self.prior_pos)
        negativeProb = math.log(self.prior_neg)
        prediction = 0

        textProcessed = preprocess_text(text)

        for word in textProcessed.split():
            positiveProb += math.log((self.pos_counter.get(word, 0) + 1) / (self.total_pos_words + self.vocab_size))
            negativeProb += math.log((self.neg_counter.get(word, 0) + 1) / (self.total_neg_words + self.vocab_size))

        if positiveProb > negativeProb:
            prediction = 1

        return prediction, positiveProb, negativeProb

