#!/usr/bin/env python

"""code template"""

import os
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
import pandas

# translation of 43 classes to 3 classes:
# 0 - prohibitory
# 1 - warning
# 2 - mandatory
# -1 - not used
class_id_to_new_class_id = {0: 0,
                            1: 0,
                            2: 0,
                            3: 0,
                            4: 0,
                            5: 0,
                            6: -1,
                            7: 0,
                            8: 0,
                            9: 0,
                            10: 0,
                            11: 1,
                            12: -1,
                            13: 1,
                            14: 0,
                            15: 0,
                            16: 0,
                            17: 0,
                            18: 1,
                            19: 1,
                            20: 1,
                            21: 1,
                            22: 1,
                            23: 1,
                            24: 1,
                            25: 1,
                            26: 1,
                            27: 1,
                            28: 1,
                            29: 1,
                            30: 1,
                            31: 1,
                            32: -1,
                            33: 2,
                            34: 2,
                            35: 2,
                            36: 2,
                            37: 2,
                            38: 2,
                            39: 2,
                            40: 2,
                            41: -1,
                            42: -1}


def load_data(path, filename):
    """
    Loads data from disk.
    @param path: Path to dataset directory.
    @param filename: Filename of csv file with information about samples.
    @return: List of dictionaries, one for every sample, with entries "image" (np.array with image) and "label" (class_id).
    """
    entry_list = pandas.read_csv(os.path.join(path, filename))

    data = []
    for idx, entry in entry_list.iterrows():
        class_id = class_id_to_new_class_id[entry['ClassId']]
        image_path = entry['Path']

        if class_id != -1:
            image = cv2.imread(os.path.join(path, image_path))
            data.append({'image': image, 'label': class_id})

    return data


def learn_bovw(data):
    """
    Learns BoVW dictionary and saves it as "voc.npy" file.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image) and "label" (class_id).
    @return: Nothing
    """
    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        kpts, desc = sift.compute(sample['image'], kpts)

        if desc is not None:
            bow.add(desc)

    vocabulary = bow.cluster()

    np.save('voc.npy', vocabulary)


def extract_features(data):
    """
    Extracts features for given data and saves it as "desc" entry.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image) and "label" (class_id).
    @return: Data with added descriptors for each sample.
    """
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()  # elementy do klasteryzacji
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)  # obliczanie konkretnych deskryptorow
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for sample in data:
        # compute descriptor and add it as "desc" entry in sample
        # TODO PUT YOUR CODE HERE
        key_points = sift.detect(sample['image'], None)  # algorytm, ktory bedzie nam szukal cech na danym obrazku
        desc = bow.compute(sample['image'], key_points)  # algorytm, ktory dla danego obrazku policzy nam deskryptory
        sample['desc'] = desc  #
        # ------------------

    return data


def train(data):
    """
    Trains Random Forest classifier.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor).
    @return: Trained model.
    """
    # train random forest model and return it from function.
    # TODO PUT YOUR CODE HERE
    # X, Y = make_classification(len(data))
    descs = []
    labels = []
    for sample in data:
        if sample['desc'] is not None:
            descs.append(
                sample['desc'].squeeze(0))  # squeeze jest po to zeby macierz 1xn zamieni nam na wektor o dlugosci n
            labels.append(sample['label'])

    clf = RandomForestClassifier()
    clf.fit(descs, labels)
    # ------------------

    return clf


def draw_grid(images, n_classes, grid_size, h, w):
    """
    Draws images on a grid, with columns corresponding to classes.
    @param images: Dictionary with images in a form of (class_id, list of np.array images).
    @param n_classes: Number of classes.
    @param grid_size: Number of samples per class.
    @param h: Height in pixels.
    @param w: Width in pixels.
    @return: Rendered image
    """
    image_all = np.zeros((h, w, 3), dtype=np.uint8)
    h_size = int(h / grid_size)
    w_size = int(w / n_classes)

    col = 0
    for class_id, class_images in images.items():
        for idx, cur_image in enumerate(class_images):
            row = idx

            if col < n_classes and row < grid_size:
                image_resized = cv2.resize(cur_image, (w_size, h_size))
                image_all[row * h_size: (row + 1) * h_size, col * w_size: (col + 1) * w_size, :] = image_resized

        col += 1

    return image_all


def predict(rf, data):
    """
    Predicts labels given a model and saves them as "label_pred" (int) entry for each sample.
    @param rf: Trained model.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor).
    @return: Data with added predicted labels for each sample.
    """
    # perform prediction using trained model and add results as "label_pred" (int) entry in sample
    # TODO PUT YOUR CODE HERE
    descs = []
    labels = []
    for idx, sample in enumerate(data):  # idx - index
        if sample['desc'] is not None:
            pred = rf.predict(sample['desc'])
            sample['label_pred'] = int(
                pred)  # rzutujemy na inta, bo dataframe jest uposledzony i dataframe nadpisuje typy zmiennych na podstawie typow pierwszych paru wartosci
    # ------------------

    return data


def evaluate(data):
    """
    Evaluates results of classification.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor), and "label_pred".
    @return: Nothing.
    """
    # evaluate classification results and print statistics
    # TODO PUT YOUR CODE HERE
    positives = 0
    negatives = 0
    pred_labels = []
    true_labels = []
    for sample in data:
        if sample['desc'] is not None:
            pred_labels.append(sample['label_pred'])
            true_labels.append(sample['label'])
            if sample['label'] == sample['label_pred']:
                positives += 1
            else:
                negatives += 1
    # ------------------
    accuracy = positives / (positives + negatives)
    print(accuracy)
    #mAP = TP/TP+FP
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    print(conf_matrix)
    # this function does not return anything
    return


def display(data):
    """
    Displays samples of correct and incorrect classification.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor), and "label_pred".
    @return: Nothing.
    """
    n_classes = 3

    corr = {}
    incorr = {}

    for idx, sample in enumerate(data):
        if sample['desc'] is not None:
            if sample['label_pred'] == sample['label']:
                if sample['label_pred'] not in corr:
                    corr[sample['label_pred']] = []
                corr[sample['label_pred']].append(idx)
            else:
                if sample['label_pred'] not in incorr:
                    incorr[sample['label_pred']] = []
                incorr[sample['label_pred']].append(idx)

            # print('ground truth = %s, predicted = %s' % (sample['label'], pred))
            # cv2.imshow('image', sample['image'])
            # cv2.waitKey()

    grid_size = 8

    # sort according to classes
    corr = dict(sorted(corr.items(), key=lambda item: item[0]))
    corr_disp = {}
    for key, samples in corr.items():
        idxs = random.sample(samples, min(grid_size, len(samples)))
        corr_disp[key] = [data[idx]['image'] for idx in idxs]
    # sort according to classes
    incorr = dict(sorted(incorr.items(), key=lambda item: item[0]))
    incorr_disp = {}
    for key, samples in incorr.items():
        idxs = random.sample(samples, min(grid_size, len(samples)))
        incorr_disp[key] = [data[idx]['image'] for idx in idxs]

    image_corr = draw_grid(corr_disp, n_classes, grid_size, 800, 600)
    image_incorr = draw_grid(incorr_disp, n_classes, grid_size, 800, 600)

    cv2.imshow('images correct', image_corr)
    cv2.imshow('images incorrect', image_incorr)
    cv2.waitKey()

    # this function does not return anything
    return


def display_dataset_stats(data):
    """
    Displays statistics about dataset in a form: class_id: number_of_samples
    @param data: List of dictionaries, one for every sample, with entry "label" (class_id).
    @return: Nothing
    """
    class_to_num = {}
    for idx, sample in enumerate(data):
        class_id = sample['label']
        if class_id not in class_to_num:
            class_to_num[class_id] = 0
        class_to_num[class_id] += 1

    class_to_num = dict(sorted(class_to_num.items(), key=lambda item: item[0]))
    # print('number of samples for each class:')
    print(class_to_num)


def balance_dataset(data, ratio):
    """
    Subsamples dataset according to ratio.
    @param data: List of samples.
    @param ratio: Ratio of samples to be returned.
    @return: Subsampled dataset.
    """
    sampled_data = random.sample(data, int(ratio * len(data)))

    return sampled_data


def main():
    data_train = load_data('./', 'Train.csv')
    print('train dataset before balancing:')
    display_dataset_stats(data_train)
    data_train = balance_dataset(data_train, 1.0)
    print('train dataset after balancing:')
    display_dataset_stats(data_train)

    data_test = load_data('./', 'Test.csv')
    print('test dataset before balancing:')
    display_dataset_stats(data_test)
    data_test = balance_dataset(data_test, 1.0)
    print('test dataset after balancing:')
    display_dataset_stats(data_test)

    # you can comment those lines after dictionary is learned and saved to disk.
    print('learning BoVW')
    learn_bovw(data_train)

    print('extracting train features')
    data_train = extract_features(data_train)

    print('training')
    rf = train(data_train)

    print('extracting test features')
    data_test = extract_features(data_test)

    print('testing on training dataset')
    data_train = predict(rf, data_train)
    evaluate(data_train)

    print('testing on testing dataset')
    data_test = predict(rf, data_test)
    evaluate(data_test)
    display(data_test)

    return


if __name__ == '__main__':
    main()

