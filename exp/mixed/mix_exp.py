import queue
import threading
import time
import argparse
import os

from emg_module import Emg_module
from acc_module import Acc_module
from data_collection import Data_collection



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="../../key_model/0906/acc/model_outlier_2_interaction")
    args = parser.parse_args()
    model_dir = args.model_dir
    data_collection = Data_collection(model_dir)
    emg_module = Emg_module(data_collection.feed_emg_data)
    acc_module = Acc_module(data_collection.feed_acc_data)
    emg_module.start()
    acc_module.start()

