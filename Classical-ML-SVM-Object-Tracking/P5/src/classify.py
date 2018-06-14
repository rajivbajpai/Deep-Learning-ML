
import numpy as np

import sys
import cv2
import glob

from color_feature import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import time
from sklearn.svm import LinearSVC
import pickle

def load_images(car_img_path, non_car_img_path):
    """
    Read the car and non car images path into the list
    """

    assert(car_img_path != None)
    assert(non_car_img_path != None)

    car_img_list = glob.glob(car_img_path + "/**/*.png", recursive=True)
    non_car_img_list = glob.glob(non_car_img_path + "/**/*.png", recursive=True)

    return (car_img_list, non_car_img_list)

def create_img_datasets(car_img_path, non_car_img_path):
    """
    Load the data and create car non car labels
    """
    car_img_list, non_car_img_list = load_images(car_img_path, non_car_img_path)
    #print(non_car_img_list)

    #read images
    car_images = []
    non_car_images = []
    labels = []
    for car_img in car_img_list:
        car = cv2.imread(car_img)
        car_images.append(car)
        labels.append(1)

    for non_car_img in non_car_img_list:
        non_car = cv2.imread(non_car_img)
        non_car_images.append(non_car)
        labels.append(-1)



    merge_img = car_images + non_car_images

    #cv2.imshow("car", merge_img[0])
    #cv2.imshow("non_car_images", merge_img[-1])
    #cv2.imshow("non_car_images1", non_car_images[-1])

    #while 1:
    #    pass

    print(non_car_images[0].shape, car_images[0].shape)
    return (np.array(car_images), np.array(non_car_images), np.array(merge_img), np.array(labels))

def get_image_features(img_database, features_cfg):

    """
    Create features vector of the training car and non car images
    """
    # please note, the img_database is 4-D array where first dimesion is the img index
    shape = img_database.shape
    print(len(shape))
    assert(len(shape) == 4)

    print(shape)
    n_images = shape[0]

    features = []

    for image_idx in range(n_images):
        img = img_database[image_idx]
        output_fvec = extract_img_features(img, features_cfg)
        features.append(output_fvec)

    #now stack one
    output_fvec = np.vstack(features).astype(np.float64)
    print(output_fvec.shape, output_fvec[0])

    # perform normalization on the input features
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(output_fvec)
    # Apply the scaler to X
    scaled_output_fvec = X_scaler.transform(output_fvec)

    print(scaled_output_fvec.shape, scaled_output_fvec[0])

    return (scaled_output_fvec, X_scaler)

def train_svc_classifier(X, y):
    """
    Linear SVM classifier
    """

    # Use a linear SVC (support vector classifier)
    svc_classifier = LinearSVC()

    #split the input into train and val images

    # shuffle the data
    X_shuffle, y_shuffle = shuffle(X, y, random_state=0)

    X_train, X_val, y_train, y_val =  train_test_split(X_shuffle, y_shuffle, test_size=0.2)

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    # Train the SVC
    svc_classifier.fit(X_train, y_train)
    accuracy_score = svc_classifier.score(X_val, y_val)
    print('Test Accuracy of SVC classifier = ', accuracy_score)

    #check some prediction through the classifier
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc_classifier.predict(X_val[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_val[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    return svc_classifier


def test_predict(svc, X, y, images):

    #check some prediction through the classifier
    X_shuffle, y_shuffle, img_shuffle = shuffle(X, y, images, random_state=0)

    t=time.time()
    n_predict = 10
    pred = svc.predict(X_shuffle[0:n_predict])
    print('My SVC predicts: ', pred)
    print('For these',n_predict, 'labels: ', y_shuffle[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    for img_idx in range(0, n_predict):
        cv2.imshow("img{0:}".format(img_idx), img_shuffle[img_idx])




def svc_predict(img, svc_classifier, normalizer):
    pass

def main(argv):
    car_img_path = argv[1]
    non_car_img_path = argv[2]
    load_classifier = int(argv[3])
    classifier_file = argv[4]

    #car_img_list, non_car_img_list = load_images(car_img_path, non_car_img_path)

    #print(len(car_img_list), len(non_car_img_list))

    cars, non_cars, merge_img, labels = create_img_datasets(car_img_path, non_car_img_path)
    print(cars.shape, non_cars.shape)
    print(merge_img.shape, labels.shape)
    print(labels[0:10], labels[-10:-2])

    features = features_vec()
    features.spatial_features = True
    features.hist_features = True
    features.hog_features = True

    features.spatial_cs = "HLS"
    features.hist_cs = "HLS"
    features.hog_cs = "HLS"

    norm_output_fvec, norm = get_image_features(merge_img, features)
    print(norm.mean_, norm.scale_, norm.var_)

    #train the classifier
    if (load_classifier == 0):
        svc_classify = train_svc_classifier(norm_output_fvec, labels)

        classifier = {}

        classifier["svc"] = svc_classify
        classifier["norm"] = norm
        print(classifier)

        #cl_file = open("./svc_classifier_new.pkl", "wb")
        cl_file = open(classifier_file,  "wb")
        pickle.dump(classifier, cl_file)
        cl_file.close()

    else:
        print("loading classifier from prev trained")
        #cl_file = open("./svc_classifier_new.pkl", "rb")
        cl_file = open(classifier_files, "rb")
        classifier = pickle.load(cl_file)
        cl_file.close()
        print(classifier)
        svc_classify = classifier["svc"]
        test_predict(svc_classify, norm_output_fvec, labels, merge_img)




    #print(car_img_list)


if __name__ == "__main__":
    main(sys.argv)
    print("End!!")
    while 1:
        pass
