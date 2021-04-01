## Python version of the developed software

### Folder structure and scripts/files explanation
* **testing/**
    * __config_tests_py.txt__ - recognition methods to test (change this file according to the methods you want to use)
    * __tiles.txt__ - tile names (__DO NOT CHANGE THIS FILE!__)
* **training/** - folder containing PY scripts for training several methods
* **remaining PY scripts**
    * __classification.py__ - contains a function for each method to perform tile classification
    * __params.py__ - contains global variables like the localization of the dataset and the trained models, and also global classes
    * __tactode_pipeline.py__ - image processing pipeline
    * __testbench.py__ - test script
