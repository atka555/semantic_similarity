import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import csv
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
from os import listdir
import json


# read data from .csv file 


CORPUS_FOLDER = "corpus"
RESULTS_FOLDER = "results"

def main():
    for f in listdir("corpus"):
        name, _ = f.split(".")

        run_calculation(name)

def run_calculation(filename: str):
    print(f"Runinng {filename}")

    samples = []
    labels = []

    with open(f"{CORPUS_FOLDER}/{filename}.csv", newline="") as read_file:
        reader = csv.reader(read_file)
        for i in list(reader)[1:]:
            samples.append(float(i[0]))
            labels.append(int(i[1]))

    X = samples
    y = labels

    trainSamples, testSamples, trainLabels, testLabels = train_test_split(X, y, test_size=0.2)
    trainSamples = np.array(trainSamples)
    testSamples = np.array(testSamples)
    trainLabels = np.array(trainLabels)
    testLabels = np.array(testLabels)
    len_x_train = len(trainSamples)
    len_x_test = len(testSamples)

    
    #  make model
    
    model = tf.keras.Sequential([
        Dense(units=16, input_shape=(1,), activation='sigmoid'),
        Dense(units=32, activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=2, activation='softmax')
    ])
    model.summary()

    
    #  train model
    

    model.compile(optimizer=Adam(learning_rate=0.0007), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=trainSamples, y=trainLabels, validation_split=0.1, batch_size=15, epochs=50, shuffle=True, verbose=2)

    
    #  evaluation
    

    loss, accuracy = model.evaluate(testSamples, testLabels)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    with open(f"{RESULTS_FOLDER}/{filename}", "w") as write_file:

            json.dump(
                {
                    "len_x_train": len_x_train,
                    "len_x_test": len_x_test,
                    "loss": str(loss),
                    "accuracy": str(accuracy),
                },
                write_file,
            )


if __name__ == "__main__":
    main()
