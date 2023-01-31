
from PIL import Image
import os
import numpy as np
import xml.etree.ElementTree as ET


def load_pictures(n):
    """load_pictures

    Args:
        n (int): Number of pictures to import from the dataset

    Returns:
        dic: A dictionnary containing the n-first pictures of the dataset (with name as key)
    """
    pictures_dic = {}
    files = os.listdir('./data/images')[:n]
    
    for file in files:
        picture = Image.open('./data/images/' + file)
        picture = np.array(picture)
        pictures_dic[file] = picture
        
    return pictures_dic

def load_annotations(n):
    """load_annotations

    Args:
        n (int): Number of pictures to import from the dataset

    Returns:
        dic: A dictionnary containing for each picture (as key) tj=he category and the list of the category of the annotated box
    """
    
    annotations_dic = {}
    files = os.listdir('./data/annotations')[:n]
    
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

def load_data(n, test = 0):
    """load_data

    Args:
        n (int): Number of pictures to import from the dataset
        test (int, optional): 1 or 0. Defaults to 0. If 0, returns all the truncated pictures, if 1 all the pictures

    Returns:
        dic: A dictionary containing for all pictures (as key) the picture saved as a numpy matrix and its category as a string 
    """
    print('Data Loading.......')
    pictures = load_pictures(n)
    annotations = load_annotations(n)

    ids = pictures.keys()
    data = {}

    print('Data Reshaping.......')
    for id in ids:
        picture = pictures[id]
        category = annotations[id]['category']
        
        if not test:
            (y_min, x_min, y_max, x_max) = annotations[id]['box']
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

def load_and_split_data(p_train = 0.7, n = 800):
    """load_and_split_data

    Args:
        p_train (float, optional): Defaults to 0.7. Percentage of the dataset that will be used for training
        n (int, optional): Number of pictures to import from the dataset. Defaults to 800
        

    Returns:
        dic: Two dictionaries containing for all pictures (as key) the picture saved as a numpy matrix and its category as a string, one for training, the other one for testing.
    """
    data = load_data(n)
    N = len(data)
    N_train = int(p_train * N)
    
    files = np.array(os.listdir('./data/annotations')[:n])
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
    """normalize_rows

    Args:
        mat (np.array): A numpy array

    Returns:
        mat: The input array with rows normalized
    """
    for i in range(len(mat)):
        norm = np.linalg.norm(mat[i, :])
        if norm != 0:
            mat[i, :] /= norm
    
    return mat
