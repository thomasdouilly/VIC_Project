# Imports

import utils
import matplotlib.pyplot as plt
import opencv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Creation of a training and a test dataset from the given pictures
train_pic, test_pic = utils.load_and_split_data(0.6, 800)

# Training of the model with the training dataset and validation with the test dataset
score, model = opencv.get_opencv_sift_model(train_pic, test_pic, 22, 8)

# Import of all the pictures from the dataset in a uncroped version
test_data = utils.load_data(800, 1)
# Building of the histograms
histo, ground_truth = opencv.get_histo_sift(test_data, 8)

# Prediction on these pictures and results
preds = model.predict(histo)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(ground_truth, preds), display_labels = ["Traffic Light", "Speed Limit", "Crosswalk", "Stop"])
cm_display.plot()
plt.show()





# Test of different values for the number of neighbours used in KMeans method
score_list = []

for k in range(1, 26):
    score, _ = opencv.get_opencv_sift_model(train_pic, test_pic, 45, 2*k)
    score_list.append(score)

plt.scatter([2*k for k in range(len(score_list))], score_list)
plt.title('Accuracy on the training set agains the size of the Bag of Features')
plt.xlabel('Size of the Bag of Features')
plt.ylabel('Accuracy of the model (in %)')
plt.show()
