import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.preprocessing import StandardScaler

data_scaler = StandardScaler()

def normalize_data(input_data):
    normalized_data = input_data.astype('float32') / 255.0
    return normalized_data

def check_data_range(data_array):
    if np.max(data_array) <= 1 and np.min(data_array) >= 0:
        return True
    else:
        return False

def inspect_label_types(labels):
    unique_label_values = np.unique(labels)
    for val in unique_label_values:
        if type(val) == 'str':
            return 'String Type'
    else:
        return 'Integers'

def calculate_model_accuracy(confusion_matrix):
    return np.diagonal(confusion_matrix).sum() / np.sum(confusion_matrix)

def filter_out_90_and_9(data: NDArray[np.floating], labels: NDArray[np.int32]):
    nine_indices = (labels == 9)
    data_90 = data[nine_indices, :]
    labels_90 = labels[nine_indices]

    data_90 = data_90[:int((data_90.shape[0]) * 0.1), :]
    labels_90 = labels_90[:int((labels_90.shape[0]) * 0.1)]

    non_nine_indices = (labels != 9)
    data_non_9 = data[non_nine_indices, :]
    labels_non_9 = labels[non_nine_indices]

    filtered_data = np.concatenate((data_non_9, data_90), axis=0)
    filtered_labels = np.concatenate((labels_non_9, labels_90), axis=0)

    return filtered_data, filtered_labels

def convert_seven_to_zero(data: NDArray[np.floating], labels: NDArray[np.int32]):
    id_seven = (labels == 7)
    id_zero = (labels == 0)
    labels[id_seven] = 0

    return data, labels

def convert_nine_to_one(data: NDArray[np.floating], labels: NDArray[np.int32]):
    id_nine = (labels == 9)
    id_one = (labels == 1)
    labels[id_nine] = 1

    return data, labels
