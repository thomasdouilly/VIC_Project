# VIC_Project

Viola-Jones part:

1) utils: load_data, load_annotation
2) creation_dataset: create the test and training dataset, dividing images in positive and negative examples for each classifiers
3) SURF: beginning of the SURF algorithm implementation with OpenCV
4) Viola-Jones alogorithm: from the data stored in positive and negative examples, we have used the Cascade-Trainer-GUI software to create the Cascade Classifiers, and then we implemented in this file the different functions of classification

Data: images (all the images), test (images kept for testing the model), XML files (contains the trained cascade classifiers), file CSCTrainer (the tentative to do it from scratch)