import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from os import listdir
import csv
import json


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
            if i[0]=='nan':
                continue
            samples.append(float(i[0]))
            labels.append(int(i[1]))

    arraysamples = np.array(samples)
    temp = arraysamples.reshape(-1, 1)
    X = temp
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    len_x_train = len(X_train)
    len_x_test = len(X_test)

    model = GaussianNB()

    with open(f"{RESULTS_FOLDER}/{filename}", "w") as write_file:

            fit = model.fit(X_train, y_train)
            score = model.score(X_test, y_test)  # accuracy

            json.dump(
                {
                    "len_x_train": len_x_train,
                    "len_x_test": len_x_test,
                    "fit": str(fit),
                    "score": str(score),
                },
                write_file,
            )


if __name__ == "__main__":
    main()
