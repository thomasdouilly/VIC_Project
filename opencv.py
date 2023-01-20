import utils
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import operator as op


pictures = utils.load_data()
def get_histo(pictures):
    
    i = 0
    des_list = []
    ref_index = [0]

    for key in pictures.keys():
        pictures[key]['picture'] = cv2.cvtColor(pictures[key]['picture'], cv2.COLOR_RGB2GRAY)

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

    histo = np.zeros((len(ref_index), 8))

    for i in range(len(ref_index) - 1):
        a, b = ref_index[i], ref_index[i + 1]
        slice = labels[a:b]
        for j in range(8):
            histo[i, j] = op.countOf(slice, j)
            
    histo = utils.normalize_rows(histo)

    return histo
neighbors_instance = KNeighborsClassifier(n_neighbors = 1)
