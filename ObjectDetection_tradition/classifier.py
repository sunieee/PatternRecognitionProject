import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import cv2

if __name__ == '__main__':
    winSize = (32, 32)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    useSignedGradients = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                            cellSize, nbins, derivAperture, winSigma, histogramNormType
                            , L2HysThreshold, gammaCorrection, nlevels, useSignedGradients)

    features = np.zeros((1, 324), np.float32)
    labels = np.zeros(1, np.int64)

    path = 'preDataset/1'
    img_files = [(os.path.join(root, name))
                 for root, dirs, files in os.walk(path)
                 for name in files]

    for i in img_files:
        img = cv2.imread(i)
        resized_img = cv2.resize(img, winSize)
        descriptor = np.transpose(hog.compute(resized_img))
        features = np.vstack((features, descriptor))
        labels = np.vstack((labels, 1))

    path = 'preDataset/0'
    img_files = [(os.path.join(root, name))
                 for root, dirs, files in os.walk(path)
                 for name in files]
    for i in img_files:
        img = cv2.imread(i)
        resized_img = cv2.resize(img, winSize)
        descriptor = np.transpose(hog.compute(resized_img))
        features = np.vstack((features, descriptor))
        labels = np.vstack((labels, 0))

    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=42)
    print('X_train: ', X_train.shape, 'y_train', y_train.shape)
    print('X_test: ', X_test.shape, 'X_test: ', y_test.shape)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))

    print('Classification report:')
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, 'aircraft_hog_svm_clf.pkl')