import os
import numpy as np

from tensorflow.compat.v1.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__)) + '/'


def load_img_4d(path):
    img = load_img(path)
    img = img_to_array(img) / 255
    img = np.expand_dims(img, 0)

    return img


def average_over_dataset(results_dict, item):
    value_list = [results_dict[file][item] for file in results_dict]
    return np.mean(value_list)
