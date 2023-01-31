import utils
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import operator as op

def get_histo_sift(pictures, numnberOfFeatures):

    # Import of pictures and categories
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
        
        # Creation of a SIFT instance
        sift_instance = cv2.SIFT_create()
        
        # Calculation of the keypoints and descriptors in all puctures from the dataset
        kp, des = sift_instance.detectAndCompute(pictures[key]['picture'],None)
        
        """
        if key == 'road34.png':
            sift = cv2.SIFT_create()
            keypoints = sift.detect(pictures[key]['picture'], None)
            output_image = cv2.drawKeypoints(pictures[key]['picture'], keypoints, 0, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
            plt.imshow(output_image)
            plt.show()
            break
        """
        # If possible, normalization of the descriptors
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

    # Creation of a K-means instance
    kmeans_instance = KMeans(numnberOfFeatures)
    
    # Feeding the K-means instance with descriptors found previously
    labels = kmeans_instance.fit(des_list).labels_

    histo = np.zeros((len(ref_index) - 1, numnberOfFeatures))


    # Count of all apprearances of each feature in all pictures
    for i in range(len(ref_index) - 1):
        a, b = ref_index[i], ref_index[i + 1]
        slice = labels[a:b]
        for j in range(numnberOfFeatures):
            histo[i, j] = op.countOf(slice, j)
    
    # Normalization of the values of the histogram
    histo = utils.normalize_rows(histo)

    return histo, np.array(category)



def get_opencv_sift_model(train_pic, test_pic, neighbors, numberOfFeatures):

    # Creation of the Kneighbors instance
    neighbors_instance = KNeighborsClassifier(n_neighbors = neighbors)
    
    # Creation of the histograms for the training and test datasets
    histo_train, categories_train = get_histo_sift(train_pic, numberOfFeatures)
    histo_test, categories_test = get_histo_sift(test_pic, numberOfFeatures)
    
    print("Model processing")
    # KNN feeding with training data
    neighbors_instance.fit(histo_train, categories_train)
    score = round(100 * neighbors_instance.score(histo_test, categories_test), 1)
    print("Score on testing dataset  :", score, "%")
    
    # KNN prediction on the test data
    preds = neighbors_instance.predict(histo_test)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(categories_test, preds), display_labels = ["Traffic Light", "Speed Limit", "Crosswalk", "Stop"])
    cm_display.plot()
    plt.show()
    
    return score, neighbors_instance