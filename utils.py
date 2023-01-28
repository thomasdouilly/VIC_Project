
from PIL import Image
import os
import numpy as np
import xml.etree.ElementTree as ET


def load_pictures():
    
    pictures_dic = {}
    files = os.listdir('./data/images')
    
    for file in files:
        picture = Image.open('./data/images/' + file)
        picture = np.array(picture)
        pictures_dic[file] = picture
        
    return pictures_dic

def load_annotations():
    
    annotations_dic = {}
    files = os.listdir('./data/annotations')
    
    for file in files:
        
        annotation = ET.parse('./data/annotations/' + file)
        root = annotation.getroot()
        
        category = root.find("./object/name").text
        
        box = root.find("./object/bndbox")
        edges = []
        for edge in box:
            edges.append(int(edge.text))
        
        features = {'category' : category, 'box' : edges}
        name = file.split('.')[0] + '.png'
        
        annotations_dic[name] = features
    
    return annotations_dic

def load_data(test = 0):
    print('Data Loading.......')
    pictures = load_pictures()
    annotations = load_annotations()

    ids = pictures.keys()
    data = {}

    print('Data Reshaping.......')
    for id in ids:
        picture = pictures[id]
        category = annotations[id]['category']
        (y_min, x_min, y_max, x_max) = annotations[id]['box']
        
        if not test:
            sign = picture[x_min : x_max + 1, y_min : y_max + 1]
        else:
            sign = picture
            
        data[id] = {"category" : category, "picture" : sign}  
    
    app_dict = {}

    for x in data.keys():
        try:
            app_dict[data[x]['category']] += 1
        except:
            app_dict[data[x]['category']] = 1
        
    print('Loaded Data Categories :', app_dict)
    
    return data

def load_and_split_data(p_train = 0.7):
    
    data = load_data()
    N = len(data)
    N_train = int(p_train * N)
    
    files = np.array(os.listdir('./data/annotations'))
    np.random.shuffle(files)
    
    data_train = {}
    data_test = {}
    
    for i in range(N):
        file = files[i]
        name = file.split('.')[0] + '.png'
        if i < N_train:
            data_train[name] = data[name]
        else:
            data_test[name] = data[name]

    return data_train, data_test


def normalize_rows(mat):
    
    for i in range(len(mat)):
        norm = np.linalg.norm(mat[i, :])
        if norm != 0:
            mat[i, :] /= norm
    
    return mat
