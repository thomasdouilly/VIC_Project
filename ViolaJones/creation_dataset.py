from utils import load_data, load_annotations
import shutil
import random
from PIL import Image
import os

def create_lists():
    """
    create data folder for training the 4 classifiers: neg and pos for each
    + Create the test dataset
    """
    data = load_data()

    key = list(data.keys())

    #remove n% for testing
    L= len(key)
    random.shuffle(key)
    N = 0.10
    lim = int(N*L)
    test = key[:lim]
    train = key[lim:]

    SL = []
    CW = []
    ST = []
    TL = []

    for pic in test :
        filename = "./data/images/" + pic
        shutil.copy(filename, "./data/test")
    
    for pic in train :
        cropped_picture = data[pic]["picture"]
        im = Image.fromarray(cropped_picture)
        filename = "./data/train/" + pic
        im.save(filename)
        
        if data[pic]["category"] == "speedlimit":
            SL.append(pic)
        elif data[pic]["category"] == "crosswalk":
            CW.append(pic)
        elif data[pic]["category"] == "stop":
            ST.append(pic)
        elif data[pic]["category"] == "trafficlight":
            TL.append(pic)

    
    for pic in SL:
        cropped_filename = "./data/train/" + pic
        basic_filename = "./data/images/" + pic
        shutil.copy(cropped_filename, "./data/speedlimitXML/p")
        shutil.copy(basic_filename, "./data/crosswalksXML/n")
        shutil.copy(basic_filename, "./data/stopXML/n")
        shutil.copy(basic_filename, "./data/trafficlightsXML/n")
    
    for pic in CW:
        cropped_filename = "./data/train/" + pic
        basic_filename = "./data/images/" + pic
        shutil.copy(basic_filename, "./data/speedlimitXML/n")
        shutil.copy(cropped_filename, "./data/crosswalksXML/p")
        shutil.copy(basic_filename, "./data/data\stopXML/n")
        shutil.copy(basic_filename, "./data/trafficlightsXML/n")
    
    for pic in ST:
        cropped_filename = "./data/train/" + pic
        basic_filename = "./data/images/" + pic
        shutil.copy(basic_filename, "./data/speedlimitXML/n")
        shutil.copy(basic_filename, "./data/crosswalksXML/n")
        shutil.copy(cropped_filename, "./data/stopXML/p")
        shutil.copy(basic_filename, "./data/trafficlightsXML/n")
    
    for pic in TL:
        cropped_filename = "./data/train/" + pic
        basic_filename = "./data/images/" + pic
        shutil.copy(basic_filename, "./data/speedlimitXML/n")
        shutil.copy(basic_filename, "./data/crosswalksXML/n")
        shutil.copy(basic_filename, "./data/stopXML/n")
        shutil.copy(cropped_filename, "./data/trafficlightsXML/p")
    
    return "Done"


#create_lists()

def keep_random_speedlimit():
    """
    remove some speedlimit signs, because the classes are too imbalanced
    """
    names = os.listdir("./data/speedlimitXML/p")
    random.shuffle(names)
    
    to_delete = names[150:]

    for elem in to_delete:
        filename = "./data/speedlimitXML/p/" + elem
        os.remove(filename)

    return "Done"

#keep_random_speedlimit()

def prepa_neg_trainer():
    """
    create the txt file with name of negative element to allow the training
    """
    with open('./data/crosswalksCSCTrainer/negative/neg.txt', 'w') as f:
        for filename in os.listdir("./data/crosswalksCSCTrainer/negative"):
            f.write('negative/'+filename+"\n")

#prepa_neg_trainer()

def prepa_pos_trainer():
    """
    same with positive element, but need to add the boxes coordinates
    """
    annotations = load_annotations()
    keys = list(annotations.keys())[0]
    with open('./data/crosswalksCSCTrainer/positive/pos.txt', 'w') as f:
        for filename in os.listdir("./data/crosswalksCSCTrainer/positive"):
            if filename != "pos.txt" :
                annotation = annotations[filename]['box']
                length = len(annotation)//4
                txt = ""
                for i in annotation:
                    txt += str(i)+" "
                f.write('positive/'+filename+"  "+str(length)+"  "+txt+"\n")

#prepa_pos_trainer()