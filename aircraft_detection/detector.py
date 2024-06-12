import cv2
import joblib
import numpy as np
from sliding_window import sliding_window, pyramid
from nms import non_max_suppression_fast

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

    clf = joblib.load('aircraft_hog_svm_clf.pkl')

    image = cv2.imread('./dataset/train/images/20__1__0___0.png')
    cv2.namedWindow('Sliding Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Sliding Window', int(image.shape[1] / 2), int(image.shape[0] / 2))

    scale = 2.0
    minSize = (500, 500)
    stepSize = 16
    (winW, winH) = (64, 64)

    bboxes = np.zeros(4, np.int64)

    for i, resized in enumerate(pyramid(image, scale=scale, minSize=minSize)):
        for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

            cropped_img = resized[y:y + winH, x:x + winW]
            cropped_img_resized = cv2.resize(cropped_img, winSize)
            descriptor = np.transpose(hog.compute(cropped_img_resized))
            y_pred = clf.predict([descriptor])

            cv2.imshow("Sliding Window", clone)
            cv2.imshow("Cropped", cropped_img)
            cv2.waitKey(1)

            if y_pred == 1:
                if i != 0:
                    bboxes = np.vstack((bboxes, np.array([
                        int(x * scale * i), int(y * scale * i),
                        int((x + winW) * scale * i), int((y + winH) * scale * i)])))
                else:
                    bboxes = np.vstack((bboxes, np.array([
                        int(x), int(y), int(x + winW), int(y + winH)])))

                print('Aircraft found!')
                cv2.waitKey(1500)

    bboxes = np.delete(bboxes, 0, axis=0)
    cv2.destroyAllWindows()

    img_bboxes = image.copy()
    for box in bboxes:
        cv2.rectangle(img_bboxes, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    cv2.namedWindow('Bounding boxes', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Bounding boxes', int(img_bboxes.shape[1] / 2), int(img_bboxes.shape[0] / 2))
    cv2.imshow('Bounding boxes', img_bboxes)

    img_nms_bboxes = image.copy()
    nms_bboxes = non_max_suppression_fast(bboxes, 0.3)

    for box in nms_bboxes:
        cv2.rectangle(img_nms_bboxes, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.namedWindow('Non maximal supression', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Non maximal supression', int(img_nms_bboxes.shape[1] / 2), int(img_nms_bboxes.shape[0] / 2))
    cv2.imshow('Non maximal supression', img_nms_bboxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
