
from PIL import Image
import os
import numpy as np
import xml.etree.ElementTree as ET


def load_pictures():
    
    data = []
    files = os.listdir('./data/images')
    
    for file in files:
        picture = Image.open('./data/images/' + file)
        picture = np.array(picture)
        picture = np.expand_dims(picture, 0)
        data.append(picture)
        
    return data

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
    
print(load_annotations())

