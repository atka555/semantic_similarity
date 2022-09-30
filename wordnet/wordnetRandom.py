# -*- coding: utf-8 -*-
import json
import string
from pprint import pprint
from nltk.tokenize import word_tokenize
import nltk
import glob
from nltk.corpus import wordnet as wn
import spacy
import itertools
import random
import numpy as np
from sklearn.utils import shuffle
import csv
from csv import reader as csv_reader


truePairs = []
randomPairs = []
with open("createPairs/englishTruePairs", "r+") as f:
    truePairsReader = csv_reader(f)

    for line in truePairsReader:
        truePairs.append(line)

with open("createPairs/englishRandomPairs", "r+") as f:
    randomPairsReader = csv_reader(f)

    for line in randomPairsReader:
        randomPairs.append(line)


'''
     http://zil.ipipan.waw.pl/SpacyPL
'''
#nlp = spacy.load('pl_spacy_model')
nlp = spacy.load("en_core_web_sm")

'''
    semSimilarity takes two sentences, tokenize them, lemmatize words, then for every word in first sentence finds most 
    similar word in second sentence; return list of max similarity for words
'''
def sentenceSimilarity (s1, s2):
    lem1 = nlp(s1)
    lem2 = nlp(s2)
    maxSimilarity = []
    for token1 in lem1:
        if wn.synsets(token1.lemma_) != []:
            wordSimilarity = []
            word1 = wn.synsets(token1.lemma_)[0]
            for token2 in lem2:
                if wn.synsets(token2.lemma_) != []:
                    word2 = wn.synsets(token2.lemma_)[0]
                    wordSimilarity.append(word1.wup_similarity(word2))
            wordSimilarityWNone = [x for x in wordSimilarity if x]
            if len(wordSimilarityWNone) != 0:
                maxSimilarity.append(max(wordSimilarityWNone))       
    return maxSimilarity
    
'''
    similarity takes two sentenes, returns sum of average value of two lists and divide this by two
'''
def similarity(s1, s2):
    max1 = sentenceSimilarity(s1, s2)
    max2 = sentenceSimilarity(s2, s1)

    if float(len(max1)) == 0 or float(len(max2)) == 0:
        return 0
    else:
        return (sum(max1) / float(len(max1)) + sum(max2) / float(len(max2))) / 2

simTruePairs = []
trueLabels = []
simRandomPairs = []
randomLabels = []

for p in truePairs:
    simTruePairs.append(similarity(p[0], p[1]))
    trueLabels.append(1)

for p in randomPairs:
    simRandomPairs.append(similarity(p[0], p[1]))
    randomLabels.append(0)

simTruePairs.extend(simRandomPairs)
trueLabels.extend(randomLabels)

trainSamples = np.array(simTruePairs)
trainLabels = np.array(trueLabels)
trainLabels, trainSamples = shuffle(trainLabels, trainSamples)

'''
    write data to .csv file
'''
myFile = open('wordnet/englishWordnetRandom.csv', 'w')
with myFile:    
    myFields = ['similarity', 'feature']
    writer = csv.DictWriter(myFile, fieldnames=myFields)    
    writer.writeheader()
    for i,j in zip(trainSamples,trainLabels):
      writer.writerow({'similarity' : i, 'feature': j})


print(sentenceSimilarity("mama", "tata"))