# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 15:06:51 2025

@author: ilyas
"""

import random
import re
from collections import defaultdict
import codecs


# ngramLM CLASS
class ngramLM:
    """Ngram Language Model Class"""

    # Create Empty ngramLM
    def __init__(self):
        self.bigramDict = defaultdict(int)
        self.unigramDict = defaultdict(int)
        self.numOfTokens = 0
        self.sizeOfVocab = 0
        self.numOfSentences = 0
        self.sentences = []
        self.totalWordCounts = 0
        self.totalBigramCount = 0
        # TO DO

    # INSTANCE METHODS
    def tokenize_line(self, line):
        # TO DO
        line = line.replace('I', 'ı').replace('İ', 'i').lower()
        pattern = r"""(?x)
                    (?:[A-ZÇĞIİÖŞÜ]\.)+
                    | \d+(?:\.\d*)?(?:'\w+)?
                    | \w+(?:-\w+)*(?:'\w+)?
                    | \.{3}
                    | [][,;.?():_!#^+$%&><|/{()=}"'\"\`-]
                    """
        tokens = re.findall(pattern, line)
        return tokens if tokens else []

    def trainFromFile(self, fn):
        # TO DO
        tokenizedSentences = []

        with open(fn, 'r', encoding='utf-8') as file:
            for line in file:
                tokens = self.tokenize_line(line.strip())
                if tokens:
                    sentence = ['<s>']
                    for token in tokens:
                        sentence.append(token)
                        if token in {'.', '?', '!'}:
                            sentence.append('</s>')
                            tokenizedSentences.append(sentence)
                            sentence = ['<s>']

                    if len(sentence) > 1:  # Line didn't end with sentence-ending token
                        sentence.append('</s>')
                        tokenizedSentences.append(sentence)

        self.sentences = tokenizedSentences
        self.numOfSentences = len(self.sentences)

        for sentence in self.sentences:
            for i, word in enumerate(sentence):
                self.unigramDict[word] += 1
                self.totalWordCounts += 1
                if i > 0:  # Count bigrams
                    bigram = (sentence[i - 1], word)
                    self.bigramDict[bigram] += 1
                    self.totalBigramCount += 1

        self.sizeOfVocab = len(self.unigramDict)

        for l in self.sentences:
            self.numOfTokens += len(l)

    def vocab(self):
        # TO DO
        l = []
        for key, val in self.unigramDict.items():
            l.append((key, val))

        l.sort(key=lambda x: (-x[1], x[0]))
        return l

    def bigrams(self):
        # TO DO
        l = []

        for key,val in self.bigramDict.items():
            l.append((key,val))

        l.sort(key=lambda x:(-x[-1],x[0]))
        return l

    def unigramCount(self, word):
        # TO DO
        if word in self.unigramDict:
            return self.unigramDict[word]
        else:
            return 0

    def bigramCount(self, bigram):
        # TO DO
        if bigram in self.bigramDict:
            return self.bigramDict[bigram]
        else:
            return 0

    def unigramProb(self, word):
        #TO DO
        #returns unsmoothed unigram probability value
        if word in self.unigramDict:
            return self.unigramDict[word] / self.totalWordCounts
        else:
            return 0

    def bigramProb(self, bigram):
        # TO DO
        # returns unsmoothed bigram probability value
        if bigram not in self.bigramDict:
            return 0
        else:
            return self.bigramDict[bigram] / self.unigramDict[bigram[0]]


    def unigramProb_SmoothingUNK(self, word):
        # TO DO
        # returns smoothed unigram probability value
        return (self.unigramDict[word] + 1) / (self.totalWordCounts + self.sizeOfVocab + 1)

    def bigramProb_SmoothingUNK(self, bigram):
        # TO DO
        # returns smoothed bigram probability value
        w1, w2 = bigram
        countW1 = self.unigramCount(w1)
        if countW1 == 0:
            return 1 / (self.sizeOfVocab + 1)
        countBigram = self.bigramCount(bigram)
        return (countBigram + 1) / (countW1 + self.sizeOfVocab + 1)


    def sentenceProb(self, sent):
        # TO DO
        # sent is a list of tokens
        # returns the probability of sent using smoothed bigram probability values
        if len(sent) == 1:  # Single word case
            return self.unigramProb_SmoothingUNK(sent[0])

        prob = 1.0
        for i in range(1, len(sent)):
            bigram = (sent[i - 1], sent[i])
            prob *= self.bigramProb_SmoothingUNK(bigram)
        return prob

    def generateSentence(self, sent=["<s>"], maxFollowWords=1, maxWordsInSent=20):
        # TO DO
        # sent is a list of tokens
        # returns the generated sentence (a list of tokens)
        while len(sent) < maxWordsInSent:
            lastWord = sent[-1]
            if lastWord == '</s>':
                break

            candidateWords = []
            for bigram, freq in self.bigramDict.items():
                if bigram[0] == lastWord:
                    nextWord = bigram[1]
                    candidateWords.append((nextWord, freq))
            candidateWords.sort(key=lambda x: (-x[1], x[0]))

            if not candidateWords:
                break

            candidateWords = candidateWords[:maxFollowWords]

            totalFrequency = 0
            for candidate in candidateWords:
                word, frequency = candidate
                totalFrequency += frequency

            randomChoice = random.randint(1, totalFrequency)
            cumulativeFrequency = 0
            for word, frequency in candidateWords:
                cumulativeFrequency += frequency
                if randomChoice <= cumulativeFrequency:
                    sent.append(word)
                    break

        return sent
