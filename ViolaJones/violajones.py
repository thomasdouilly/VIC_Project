import cv2
import os
import matplotlib.pyplot as plt
from utils import load_annotations
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
import numpy as np
from skimage.transform import integral_image
from skimage.feature import haar_like_feature, haar_like_feature_coord, draw_haar_like_feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


def hasard():
    """
    return the 10 first image with object detected of speedlimitClassifier
    """
    box = ()
    made = 0
    i=0
    names = os.listdir('./data/images')
    classifier = cv2.CascadeClassifier("./data/speedlimitXML/classifier/cascade.xml")
    L = len(names)
    while made<10:
        name = names[i]
        filepath = "./data/images/" + name
        image_to_compare_with = cv2.imread(filepath)
        image_to_compare_with_resized = cv2.resize(image_to_compare_with, (450, 450))
        image_to_compare_with_grayscale = cv2.cvtColor(image_to_compare_with_resized, cv2.COLOR_BGR2GRAY)
        box = classifier.detectMultiScale(image_to_compare_with_grayscale)
        i+=1
        if box != ():
            made+=1
            for(x,y,w,h) in box:
                resized=cv2.rectangle(image_to_compare_with_resized,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.imshow('img',resized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    return "Done"

def choix(i):
    """
    return only a specified element of index i
    """
    classifier = cv2.CascadeClassifier("./data/speedlimitXML/classifier/cascade.xml")
    #names = os.listdir('./data/images')
    #name = names[i]
    name = "road110.png"
    filepath = "./data/images/" + name
    image_to_compare_with = cv2.imread(filepath)
    image_to_compare_with_resized = cv2.resize(image_to_compare_with, (450, 450))
    image_to_compare_with_grayscale = cv2.cvtColor(image_to_compare_with_resized, cv2.COLOR_BGR2GRAY)
    box = classifier.detectMultiScale(image_to_compare_with_grayscale)
    if box != ():
        for(x,y,w,h) in box:
            image_to_compare_with_resized=cv2.rectangle(image_to_compare_with_resized,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('img',image_to_compare_with_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return "Done"

hasard()
#choix(1)

def classification_test(folder, name):
    """
    compute the test on the image of the folder
    return the predicted class, decided with the number of boxes detected
    in case of equality between classes, the choice is done randomly
    """
    classif = {}

    classifierCW = cv2.CascadeClassifier("./data/crosswalksXML/classifier/cascade.xml")
    classifierSL = cv2.CascadeClassifier("./data/speedlimitXML/classifier/cascade.xml")
    classifierST = cv2.CascadeClassifier("./data/stopXML/classifier/cascade.xml")
    classifierTL = cv2.CascadeClassifier("./data/trafficlightsXML/classifier/cascade.xml")
    
    filepath = "./data/" + folder + "/"+ name
    image_to_compare_with = cv2.imread(filepath)
    image_to_compare_with_resized = cv2.resize(image_to_compare_with, (450, 450))
    image_to_compare_with_grayscale = cv2.cvtColor(image_to_compare_with_resized, cv2.COLOR_BGR2GRAY)
    boxCW = classifierCW.detectMultiScale(image_to_compare_with_grayscale)
    boxSL = classifierSL.detectMultiScale(image_to_compare_with_grayscale)
    boxST = classifierST.detectMultiScale(image_to_compare_with_grayscale)
    boxTL = classifierTL.detectMultiScale(image_to_compare_with_grayscale)
    
    classif["crosswalk"] = len(boxCW)
    classif["speedlimit"] = len(boxSL)
    classif["stop"] = len(boxST)
    classif["trafficlight"] = len(boxTL)
    new_classif = sorted(classif.items(), key=lambda x: x[1], reverse = True)
    new_classif = dict(new_classif)
    keys = list(new_classif.keys())
    if new_classif[keys[0]] > new_classif[keys[1]]:
        res = keys[0]
    elif new_classif[keys[1]] > new_classif[keys[2]]:
        i = random.randint(0,1)
        res = keys[i]
    elif new_classif[keys[2]] > new_classif[keys[3]]:
        i = random.randint(0,2)
        res = keys[i]
    else : 
        res = "speedlimit"
    return res

#print(classification_test("test", "road5.png"))

def test():
    """
    create the y_pred and y_test variables on the test dataset
    """
    y_pred = []
    y_test = []
    print("load_annotations...")
    annotations = load_annotations()
    names = os.listdir("./data/test")
    print("classification...")
    for name in names:
        y_test.append(annotations[name]["category"])
        y_pred.append(classification_test("test", name))

    return y_pred, y_test


def results():
    """
    create confusion matrix and classification report
    """
    y_pred, y_test = test()

    CM = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(CM)
    sn.heatmap(df_cm, annot=True)
    plt.show()

    print(classification_report(y_test, y_pred))

    return 'Done'

#results()

### IMPLEMENTATION FROM SCRATCH

#take a picture
#keep only the interest region
#resize 25x25
#set to grayscale
#store in images variables

images = []
list_dir1 = os.listdir("./data/crosswalksXML/p")
list_dir2 = os.listdir("./data/crosswalksXML/n_2")
w = 25
h = 25
for name in list_dir1 :
    filepath = "./data/crosswalksXML/p/" + name
    image_to_compare_with = cv2.imread(filepath)
    image_to_compare_with_resized = cv2.resize(image_to_compare_with, (w, h))
    image_to_compare_with_grayscale = cv2.cvtColor(image_to_compare_with_resized, cv2.COLOR_BGR2GRAY)
    images.append(image_to_compare_with_grayscale)
for name in list_dir2:
    filepath = "./data/crosswalksXML/n_2/" + name
    image_to_compare_with = cv2.imread(filepath)
    image_to_compare_with_resized = cv2.resize(image_to_compare_with, (w, h))
    image_to_compare_with_grayscale = cv2.cvtColor(image_to_compare_with_resized, cv2.COLOR_BGR2GRAY)
    images.append(image_to_compare_with_grayscale)


def extract_feature_image(img):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1])

#print(extract_feature_image(image_to_compare_with_grayscale))

def features_all_images():
    X = [extract_feature_image(img) for img in images]
    X = np.array(X)
    L = len(images)//2
    y = np.array([1] * L + [0] * L)
    return X, y

#print(features_all_images())

def training(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2,
                                                    random_state=0,
                                                    stratify=y)
    feature_coord, feature_type = haar_like_feature_coord(width=w, height=h)
    
    clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                             max_features=100, n_jobs=-1, random_state=0)

    clf.fit(X_train, y_train)
    auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(auc_full_features)
    # Sort features in order of importance and plot the six most significant
    idx_sorted = np.argsort(clf.feature_importances_)[::-1]

    return feature_coord, idx_sorted

def plot_features(feature_coord, idx_sorted):
    fig, axes = plt.subplots(3, 2)
    for idx, ax in enumerate(axes.ravel()):
        image = images[1]
        image = draw_haar_like_feature(image, 0, 0,h, w, [feature_coord[idx_sorted[idx]]])
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])

    _ = fig.suptitle('The most important features')
    plt.show()

#print("extraction...")
#X, y = features_all_images()
#print("extraction done")
#feature_coord, idx_sorted = training(X, y)
#print("model trained")
#plot_features(feature_coord, idx_sorted)