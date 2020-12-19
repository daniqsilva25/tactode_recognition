## Python version of the developed software

### Folder structure and scripts/files explanation
* **testing/**
    * __config_tests_py.txt__ - recognition methods to test (change this file according to the methods you want to use)
    * __tiles.txt__ - tile names (__DO NOT CHANGE THIS FILE!__)
* **tools/**
    * ___annotate_dataset.py___ - script for performing annotations on the training dataset (this is a fundamental step to train _YOLOv4_ and _SSD_)
    * ___ged_data.py___ - script for cropping and resizing the images in the training dataset (this is a fundamental step to train _VGG16_, _VGG19_ and _MobileNetV2_)
    * ___run_yolo_opencv.py___ - example script for running _YOLO_ with _OpenCV_
* **training/** - folder containing _.py_ scripts for training several methods
* **remaining \*.py scripts**
    * ___classification___ - contains a function for each method to perform tile classification
    * ___params___ - contains global variables like the localization of the dataset and the trained models, and also global classes
    * ___tactode_pipeline___ - image processing pipeline
    * ___testbench___ - test script
