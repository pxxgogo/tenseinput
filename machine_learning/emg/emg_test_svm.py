import numpy as np
from sklearn import svm
import argparse
import os
TRAINABLE_NUM = 1000


def generate_trainable_data(data_dir):
    raw_data = np.load(data_dir)
    return np.array(list(raw_data[:TRAINABLE_NUM, 0])), np.array(list(raw_data[:TRAINABLE_NUM, 1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)

    args = parser.parse_args()
    data_dir = args.data_dir
    training_data_dir = os.path.join(data_dir, "training_data.npy")
    test_data_dir = os.path.join(data_dir, "test_data.npy")
    train_x, train_y = generate_trainable_data(training_data_dir)
    test_x, test_y = generate_trainable_data(test_data_dir)
    print(train_x.shape)
    print(train_y.shape)
    clf = svm.SVC(kernel='linear', C=2)
    clf.fit(train_x, train_y)
    print(clf.score(test_x, test_y))

main()