from sklearn import feature_extraction
import time
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import cv2
from utils import load_annotations, load_data
import random


def load_images():
    """
    :param base: path to the folder containing the images
    :param classes: names the the folders
    :return: a list, containing key value pairs (image, class)
    """
    pictures = load_data()
    classes = load_annotations()
    ids = pictures.keys()

    labelled_images = [(pictures[id], classes[id]) for id in ids]

    return labelled_images


print("Load_data")
data = load_images()
random.shuffle(data)
N = len(data)
train_size = 0.8
index_lim = int(train_size*N)
train_set, test_set = load_images()[:index_lim], load_images()[index_lim:]

# extract features from training images
print("Using SURF to extract features from the images...")

train_descriptors, train_labels = feature_extraction.get_descriptors_and_labels(feature_extraction.apply_surf, train_set)

print("Training the model...")
start = time.time()

svm = LinearSVC()
svm.fit(train_descriptors, train_labels)

end = time.time()

print("Training the model took " + str(end - start) + " seconds.")

# extract features from test images
test_descriptors, test_labels = feature_extraction.get_descriptors_and_labels(feature_extraction.apply_surf, test_set)

# test the model and print report
predictions = svm.predict(test_descriptors)
print(classification_report(test_labels, predictions))