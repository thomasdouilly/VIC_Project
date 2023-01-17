
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

def load_data():
    pictures = load_pictures()
    annotations = load_annotations()

    ids = pictures.keys()
    data = {}

    for id in ids:
        picture = pictures[id]
        category = annotations[id]['category']
        (y_min, x_min, y_max, x_max) = annotations[id]['box']

        sign = picture[x_min : x_max + 1, y_min : y_max + 1]
        
        data[id] = {"category" : category, "picture" : sign}  
    
    return data