# Tactode tiles recognition
This repository holds the developed software aiming Tactode tiles recognition without ArUco markers.

## Tactode - _Tactile Coding_
Tactode is a tangible programming system whose main goal is the earlier development of children's programming skills and computational thinking. This system is composed by a web application (with a simulator), puzzle-like tiles and a real robot. The children build a Tactode code using the tiles, take a picture of it and then they can upload it to the application to be tested by the simulator and, possibly, executed later on the robot.
Initially, Tactode relied on ArUco markers to perform the recognition of the tiles, but with this work the markers were removed and the tiles are now recognized by their own content using: (i) machine learning with HOG&SVM, (ii) deep learning with neural networks such as VGG16/17, MobileNetV2, YOLOv4 and SSD, (iii) matching of handcrafted features with SIFT, SURF, ORB and BRISK, and (iv) template matching.

**NOTE:** if you want to know more about Tactode check the [official website](https://fe.up.pt/asousa/tactode).

## Repository structure
* **dataset/** - datasets for testing all the methods, training learning-based methods and, also a template dataset for using with Template Matching and Features Detection and Description methods
* **trained_models/** - trained models that were tested targeting Tactode tiles recognition
* **tactode-python/** - Python version of the developed software
* **tactode-nodejs/** - JavaScript/Node.js version of the developed software

## Software dependencies
* **Node.js**
    * [opencv4nodejs](https://www.npmjs.com/package/opencv4nodejs)

* **Python**
    * [opencv-contrib-python](https://pypi.org/project/opencv-contrib-python/)
    * [tensorflow](https://www.tensorflow.org/install)

