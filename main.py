import utils
import matplotlib.pyplot as plt
import opencv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


train_pic, test_pic = utils.load_and_split_data(0.6)
score, model = opencv.get_opencv_sift_model(train_pic, test_pic, 22)

test_data = utils.load_data(1)
histo, ground_truth = opencv.get_histo_sift(test_data)
preds = model.predict(histo)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(ground_truth, preds), display_labels = ["Traffic Light", "Speed Limit", "Crosswalk", "Stop"])
cm_display.plot()
plt.show()

"""
score_list = []

for k in range(30):
    score = opencv.get_opencv_sift_model(train_pic, test_pic, 2*k + 1)
    score_list.append(score)
    
plt.scatter([2*k+1 for k in range(30)], score_list)
plt.title('Accuracy on the training set agains the number of neighbours per iteration')
plt.xlabel('Number of neighbours')
plt.ylabel('Accuracy of the model (in %)')
plt.show()
"""