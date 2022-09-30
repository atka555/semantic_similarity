from absl import logging

import tensorflow as tf
import tensorflow_text

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import pprint
import csv
from csv import reader as csv_reader
from sklearn.utils import shuffle

module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)

truePairs = []
randomPairs = []
with open("pairs/cutietestrun4June2020TruePairs", "r+") as f:
    truePairsReader = csv_reader(f)

    for line in truePairsReader:
        truePairs.append(line)

with open("pairs/cutietestrun4June2020RandomPairs", "r+") as f:
    randomPairsReader = csv_reader(f)

    for line in randomPairsReader:
        randomPairs.append(line)


print(truePairs[0])
messages = [truePairs[0][0], truePairs[0][1]]

trueSimList = []
trueLabels = []
randomSimList = []
randomLabels = []

for pair in truePairs:
    messages = [pair[0], pair[1]]
    message_embeddings = embed(messages)
    corr = np.inner(message_embeddings, message_embeddings)
    trueSimList.append(corr[0][1])
    trueLabels.append(1)

for pair in randomPairs:
    messages = [pair[0], pair[1]]
    message_embeddings = embed(messages)
    corr = np.inner(message_embeddings, message_embeddings)
    randomSimList.append(corr[0][1])
    randomLabels.append(0)

trueSimList.extend(randomSimList)
trueLabels.extend(randomLabels)

trainSamples = np.array(trueSimList)
trainLabels = np.array(trueLabels)
trainLabels, trainSamples = shuffle(trainLabels, trainSamples)

'''
    write data to .csv file
'''
myFile = open('cutietestrun4June2020SentenceEmbeddingRandom.csv', 'w')
with myFile:    
    myFields = ['similarity', 'feature']
    writer = csv.DictWriter(myFile, fieldnames=myFields)    
    writer.writeheader()
    for i,j in zip(trainSamples,trainLabels):
      writer.writerow({'similarity' : i, 'feature': j})