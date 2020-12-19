import time

# GLOBAL VARIABLES
TESTS = {
    'config': 'testing/config_tests_py.txt',
    'tiles': 'testing/tiles.txt'
}

RESULTS = {
    'folder': '../../results/',
    'file': 'results.txt'
}

DATASET = {
    'templates':  '../dataset/templates',
    'test_small': '../dataset/test/tactode_small',
    'test_big':   '../dataset/test/tactode_big',
    'train_cla':  '../dataset/train/tile_classification',
    'train_det':  '../dataset/train/teeth_detection'
}


# CLASSES
class ConfMatrix:
    def __init__(self, name=""):
        self.id = name
        self.count = 0


class Confusions:
    def __init__(self, name=""):
        self.id = name
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.conf_matrix = []


class PieceError:
    def __init__(self, row=-1, col=-1, expected_p="", obtained_p=""):
        self.i = row
        self.j = col
        self.expected_piece = expected_p
        self.obtained_piece = obtained_p


class ErrorsList:
    def __init__(self, lst=[], n_err=-1):
        self.list = lst
        self.num_errors = n_err


class PiecesList:
    def __init__(self, lst=[], n_pieces=-1, rot=False):
        self.list = lst
        self.num_pieces = n_pieces
        self.is_rotated = rot


class Size:
    def __init__(self, w=-1, h=-1):
        self.width = w
        self.height = h


class HogSvmDetector:
    def __init__(self):
        self.svm_file = "../trained_models/hogsvm_PY/svm_detector_PY.yaml"
        self.hog_file = "../trained_models/hogsvm_PY/hog_detector.yaml"
        self.size = Size(w=64, h=32)


class HogSvmClassifier:
    def __init__(self):
        self.classes_file = "../trained_models/hogsvm_PY/pieces.txt"
        self.svm_file = "../trained_models/hogsvm_PY/svm_classifier_PY.yaml"
        self.hog_file = "../trained_models/hogsvm_PY/hog_classifier.yaml"
        self.size = Size(w=32, h=32)
        self.classes = ""

    def load_classes(self):
        with open(self.classes_file, "r") as f:
            self.classes = f.readlines()


class YOLO:
    def __init__(self, obj_thresh=.1, conf_thresh=.1, nms_thresh=.1):
        self.classes_file = "../trained_models/yolo/obj.names"
        self.cfg_file = "../trained_models/yolo/yolov4-obj.cfg"
        self.weights_file = "../trained_models/yolo/yolov4-obj_last.weights"
        self.size = Size(w=416, h=416)
        self.object_threshold = obj_thresh  # Objects threshold
        self.conf_threshold = conf_thresh   # Confidence threshold
        self.nms_threshold = nms_thresh     # Non-maximum suppression threshold
        self.classes = ""

    def load_classes(self):
        with open(self.classes_file, "rt") as f:
            self.classes = f.read().rstrip("\n").split("\n")


class SSD:
    def __init__(self, obj_thresh=.1, conf_thresh=.1, nms_thresh=.1):
        self.classes_file = "../trained_models/ssd/classes.txt"
        self.model_file = "../trained_models/ssd/frozen_inference_graph.pb"
        self.size = Size(w=300, h=300)
        self.object_threshold = obj_thresh
        self.conf_threshold = conf_thresh
        self.nms_threshold = nms_thresh
        self.classes = ""

    def load_classes(self):
        with open(self.classes_file, "rt") as f:
            self.classes = f.read().rstrip("\n").split("\n")


class VGG16:
    def __init__(self):
        self.classes_file = "../trained_models/vgg16/classes.txt"
        self.model_folder = "../trained_models/vgg16/finetuned"
        self.size = Size(w=224, h=224)
        self.classes = ""

    def load_classes(self):
        with open(self.classes_file, "rt") as f:
            self.classes = f.read().rstrip("\n").split("\n")


class VGG19:
    def __init__(self):
        self.classes_file = "../trained_models/vgg19/classes.txt"
        self.model_folder = "../trained_models/vgg19/finetuned"
        self.size = Size(w=224, h=224)
        self.classes = ""

    def load_classes(self):
        with open(self.classes_file, "rt") as f:
            self.classes = f.read().rstrip("\n").split("\n")


class MobileNetV2:
    def __init__(self):
        self.classes_file = "../trained_models/mobilenetV2/classes.txt"
        self.model_folder = "../trained_models/mobilenetV2/finetuned"
        self.size = Size(w=224, h=224)
        self.classes = ""

    def load_classes(self):
        with open(self.classes_file, "rt") as f:
            self.classes = f.read().rstrip("\n").split("\n")


class TemplateMatchFeatDetDesc:
    def __init__(self):
        self.folder = DATASET['templates']
        self.size = Size(w=350, h=350)


# FUNCTIONS
def get_current_time():
    return time.strftime("%d/%m/%Y %H:%M:%S", time.localtime())
