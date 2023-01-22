import utils
import matplotlib.pyplot as plt
import opencv

train_pic, test_pic = utils.load_and_split_data(0.8)
opencv.get_opencv_sift_model(train_pic, test_pic, 2)
opencv.get_opencv_surf_model(train_pic, test_pic, 2)

"""
score_list = []

for k in range(30):
    score = opencv.get_opencv_sift_model(train_pic, test_pic, 2*k + 1)
    score_list.append(score)
    
plt.scatter([2*k+1 for k in range(30)], score_list)
plt.show()
"""