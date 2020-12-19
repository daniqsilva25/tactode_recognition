import numpy as np
import os
import random
import cv2 as cv
import params


# CLASSES
class Contour:
    def __init__(self, idx=-1, area=-1):
        self.idx = idx
        self.area = area


class Rect:
    def __init__(self, xx=-1, yy=-1, hc=-1, wc=-1):
        self.x = int(xx)
        self.y = int(yy)
        self.height = int(hc)
        self.width = int(wc)

    def to_square(self):
        diff = int(round(abs(self.width - self.height) / 2))
        side = self.width if self.width >= self.height else self.height
        self.x = self.x if self.width >= self.height else self.x - diff
        self.y = self.y if self.height >= self.width else self.y - diff
        self.height, self.width = side, side


# FUNCTIONS
def cut_piece(img_path=""):
    cnt_arr = []
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    a, bw = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(cv.UMat(bw), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        cnt = contours[i]
        cnt_arr.append(Contour(i, cv.contourArea(cnt)))
    cnt_arr.sort(key=lambda c: c.area, reverse=True)
    x, y, w, h = cv.boundingRect(contours[cnt_arr[1].idx])
    h = round(h / 3)
    y = y + 2 * h
    h = round(w / 2)
    bbox = Rect(x, y, h, w)
    return img[bbox.y: bbox.y + bbox.height, bbox.x: bbox.x + bbox.width]


def compute_hog(roi, hog, config):
    roi = cv.resize(roi, (config.size.width, config.size.height), cv.INTER_AREA)
    cv.imshow("roi", cv.resize(roi, (-1, -1), fx=2, fy=2))
    cv.waitKey(1)
    hist = hog.compute(roi)
    return hist


# main
print("training HOG&SVM based detector started: %s" % params.get_current_time())

detection_dataset = params.DetectionDataset()
hog_svm_config = params.HogSvmDetector()

hog = cv.HOGDescriptor(hog_svm_config.hog_file)
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.5)

samples, labels = [], []
for p_type in sorted(os.listdir(detection_dataset.train)):
    print("\n > loading '%s' pieces ..." % p_type)
    type_folder = os.path.join(detection_dataset.train, p_type)
    label = 1 if p_type == "positive" else -1
    for p_name in sorted(os.listdir(type_folder)):
        print("    * computing descriptors for '%s' ..." % p_name)
        piece_folder = os.path.join(type_folder, p_name)
        for img in sorted(os.listdir(piece_folder)):
            img_path = os.path.join(piece_folder, img)
            desc = compute_hog(cut_piece(img_path), hog, hog_svm_config)
            samples.append(desc)
            labels.append(label)
cv.destroyAllWindows()

# Fisher-Yates shuffle algorithm
print("\n > shuffling lists ...")
for i in range(len(labels) - 1, 0, -1):
    j = random.randint(0, i + 1)
    labels[i], labels[j] = labels[j], labels[i]
    samples[i], samples[j] = samples[j], samples[i]

x_train = np.array(samples, dtype=np.float32)
y_train = np.array(labels, dtype=np.int)
print("\n > training ...")
ret = cv.ml_SVM.trainAuto(svm, x_train, cv.ml.ROW_SAMPLE, y_train, kFold=10)
svm.save("svm_detector_py.yaml")

print("\ntraining HOG&SVM based detector ended:   %s" % params.get_current_time())
