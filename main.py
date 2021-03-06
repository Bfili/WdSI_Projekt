import os
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET


def load_data(path):

    data = []
    info_data = []
    os.chdir(r".\dataset\annotations")
    for subdir, dirs, files in os.walk('.', topdown=False):
        for file in files:
            print(file)
            tree = ET.parse(os.path.join(subdir, file))
            root = tree.getroot()
            class_id = ''
            info_data_dict = {}
            image_file_name = root.find('filename').text
            info_data_dict = {'filename': image_file_name,
                              'number_of_objects': len(root.findall('object')),
                              'coordinates': []}
            height = 0
            width = 0
            for child in root.findall('size'):
                height = child.find('height').text
                width = child.find('width').text

            for child in root.findall('object'):
                class_id = child.find('name').text
                X1, X2, Y1, Y2 = 0, 0, 0, 0
                for coordinate in child.findall('bndbox'):
                    X1 = coordinate.find('xmin').text
                    X2 = coordinate.find('xmax').text
                    Y1 = coordinate.find('ymin').text
                    Y2 = coordinate.find('ymax').text
                object_width = float(X2) - float(X1)
                object_height = float(Y2) - float(Y1)
                if object_width > 0.1*float(width) and object_height > 0.1*float(height):
                    info_data_dict['coordinates'].append([X1, X2, Y1, Y2])
                    os.chdir(r"..\images")
                    image_path = r'{}'.format(image_file_name)
                    image = cv2.imread(os.path.join(path, image_path))
                    cropped_im = image[int(Y1):int(Y2), int(X1):int(X2)]
                    num_class_id = 0
                    if class_id == 'crosswalk':
                        num_class_id = 1
                    data.append({'image': cropped_im, 'label': num_class_id})
                    info_data.append(info_data_dict)
                    os.chdir(r"..\annotations")
                else:
                    info_data_dict['number_of_objects'] -= 1
    os.chdir(r"..\..")
    return data, info_data


def print_test_data(data):

    for dict in data:
        print(dict['filename'])
        print(dict['number_of_objects'])
        for xy in dict['coordinates']:
            for element in xy:
                print(element, end=" ")
            print()
    return

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
        key_points = sift.detect(sample['image'], None)  # algorytm, ktory bedzie nam szukal cech na danym obrazku
        desc = bow.compute(sample['image'], key_points)  # algorytm, ktory dla danego obrazku policzy nam deskryptory
        sample['desc'] = desc
        # ------------------

    return data


def train(data):
    """
    Trains Random Forest classifier.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor).
    @return: Trained model.
    """
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
    for idx, sample in enumerate(data):  # idx - index
        if sample['desc'] is not None:
            pred = rf.predict(sample['desc'])
            sample['label_pred'] = pred
    # ------------------

    return data


def evaluate(data):
    """
    Evaluates results of classification.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor), and "label_pred".
    @return: Nothing.
    """
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
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    print(conf_matrix)
    avg_precision = conf_matrix[1][1]/(conf_matrix[1][1]+conf_matrix[0][1])
    print("Average precision: ", avg_precision)
    avg_recall = conf_matrix[1][1]/(conf_matrix[1][1]+conf_matrix[1][0])
    print("Recall: ", avg_recall)

    return


def display(data):
    """
    Displays samples of correct and incorrect classification.
    @param data: List of dictionaries, one for every sample, with entries "image" (np.array with image), "label" (class_id),
                    "desc" (np.array with descriptor), and "label_pred".
    @return: Nothing.
    """
    n_classes = 2

    corr = {}
    incorr = {}

    for idx, sample in enumerate(data):
        if sample['desc'] is not None:
            if sample['label_pred'] == sample['label']:
                if sample['label_pred'].item() not in corr:
                    corr[sample['label_pred'].item()] = []
                corr[sample['label_pred'].item()].append(idx)
            else:
                if sample['label_pred'].item() not in incorr:
                    incorr[sample['label_pred'].item()] = []
                incorr[sample['label_pred'].item()].append(idx)

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

    return


def display_dataset_stats(data):
    """
    Displays statistics about dataset in a form: class_id: number_of_samples
    @param data: List of dictionaries, one for every sample, with entry "label" (class_id).
    @return: Nothing
    """
    class_to_num = {'crosswalk': 0,
                    "other": 0}
    for idx, sample in enumerate(data):
        class_id = sample['label']
        if class_id == 1:
            class_to_num['crosswalk'] +=1
        else:
            class_to_num['other'] +=1

    class_to_num = dict(sorted(class_to_num.items(), key=lambda item: item[0]))
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
    data, info_data = load_data('./')
    print('dataset before balancing:')
    display_dataset_stats(data)
    data = balance_dataset(data, 1.0)
    print('dataset after balancing:')
    display_dataset_stats(data)
    data_train, data_test, info_data_train, info_data_test = train_test_split(data, info_data, test_size=0.3, random_state=42)

    print_test_data(info_data_test)

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

