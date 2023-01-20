import cv2
import os
import matplotlib.pyplot as plt
from utils import load_annotations
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd

def hasard():
    box = ()
    made = 0
    i=0
    names = os.listdir('./data/images')
    classifier = cv2.CascadeClassifier(r"C:\Users\antoi\OneDrive\Bureau\CS\3A\SDI\VIC\Project\git\VIC_Project\data\speedlimitXML\classifier\cascade.xml")
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
    classifier = cv2.CascadeClassifier(r"C:\Users\antoi\OneDrive\Bureau\CS\3A\SDI\VIC\Project\git\VIC_Project\data\speedlimitXML\classifier\cascade.xml")
    #names = os.listdir('./data/images')
    #name = names[i]
    name = "road5.png"
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

#hasard()
#choix(1)

def classification_test(name):
    classif = []

    classifierCW = cv2.CascadeClassifier(r"C:\Users\antoi\OneDrive\Bureau\CS\3A\SDI\VIC\Project\git\VIC_Project\data\crosswalksXML\classifier\cascade.xml")
    classifierSL = cv2.CascadeClassifier(r"C:\Users\antoi\OneDrive\Bureau\CS\3A\SDI\VIC\Project\git\VIC_Project\data\speedlimitXML\classifier\cascade.xml")
    classifierST = cv2.CascadeClassifier(r"C:\Users\antoi\OneDrive\Bureau\CS\3A\SDI\VIC\Project\git\VIC_Project\data\stopXML\classifier\cascade.xml")
    classifierTL = cv2.CascadeClassifier(r"C:\Users\antoi\OneDrive\Bureau\CS\3A\SDI\VIC\Project\git\VIC_Project\data\trafficlightsXML\classifier\cascade.xml")
    
    filepath = "./data/test/" + name
    image_to_compare_with = cv2.imread(filepath)
    image_to_compare_with_resized = cv2.resize(image_to_compare_with, (450, 450))
    image_to_compare_with_grayscale = cv2.cvtColor(image_to_compare_with_resized, cv2.COLOR_BGR2GRAY)
    boxCW = classifierCW.detectMultiScale(image_to_compare_with_grayscale)
    boxSL = classifierSL.detectMultiScale(image_to_compare_with_grayscale)
    boxST = classifierST.detectMultiScale(image_to_compare_with_grayscale)
    boxTL = classifierTL.detectMultiScale(image_to_compare_with_grayscale)
    
    if len(boxCW) >0:
        classif.append("crosswalk")
    if len(boxSL) >0:
        classif.append("speedlimit")
    if len(boxST) >0:
        classif.append("stop")
    if len(boxTL) >0:
        classif.append("trafficlight")

    if len(classif)>1:
        i = random.randint(0,len(classif)-1)
        classif = classif[i]

    if len(classif) == 0:
        classif = "no_roadsigns_detected"

    if len(classif) == 1:
        classif = classif[0]

    return classif


def test():
    y_pred = []
    y_test = []
    annotations = load_annotations()
    names = os.listdir("./data/test")
    for name in names:
        y_test.append(annotations[name]["category"])
        pred = classification_test(name)
        y_pred.append(pred)

    return y_pred, y_test

y_pred, y_test = test()

CM = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(CM)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()

print(classification_report(y_test, y_pred))