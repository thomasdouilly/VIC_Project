import utils
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import operator as op

def get_histo_sift(pictures):
    
    i = 0
    des_list = []
    category = []
    ref_index = [0]

    for key in pictures.keys():
        try:
            pictures[key]['picture'] = cv2.cvtColor(pictures[key]['picture'], cv2.COLOR_RGB2GRAY)
        except:
            0
            
        category.append(pictures[key]['category'])
        sift_instance = cv2.SIFT_create()
        kp, des = sift_instance.detectAndCompute(pictures[key]['picture'],None)

        try:
            des = utils.normalize_rows(des)
            des = np.where(des > 0.2, 0.2, des)
            des = utils.normalize_rows(des)
            des_list.append(des)
            ref_index.append(ref_index[-1] + len(des))
        except:
            print('No features extracted from : ' + key)
            ref_index.append(ref_index[-1])

    des_list = np.concatenate(des_list, axis = 0)

    kmeans_instance = KMeans()
    labels = kmeans_instance.fit(des_list).labels_

    histo = np.zeros((len(ref_index) - 1, 8))

    for i in range(len(ref_index) - 1):
        a, b = ref_index[i], ref_index[i + 1]
        slice = labels[a:b]
        for j in range(8):
            histo[i, j] = op.countOf(slice, j)
            
    histo = utils.normalize_rows(histo)

    return histo, np.array(category)



def get_opencv_sift_model(train_pic, test_pic, neighbors):
    
    neighbors_instance = KNeighborsClassifier(n_neighbors = neighbors)
    
    histo_train, categories_train = get_histo_sift(train_pic)
    histo_test, categories_test = get_histo_sift(test_pic)
    print("Model processing")
    neighbors_instance.fit(histo_train, categories_train)
    score = round(100 * neighbors_instance.score(histo_test, categories_test), 1)
    print("Score on testing dataset  :", score, "%")
    preds = neighbors_instance.predict(histo_test)
    print("Confusion matrix :")
    print(confusion_matrix(categories_test, preds))
    return score
        