## Python version of the developed software

### Folder structure and scripts/files explanation
* **testing/**
    * __config_tests_py.txt__ - recognition methods to test (change this file according to the methods you want to use)
    * __tiles.txt__ - tile names (__DO NOT CHANGE THIS FILE!__)
* **training/** - folder containing _.py_ scripts for training several methods
* **remaining \*.py scripts**
    * ___classification___ - contains a function for each method to perform tile classification
    * ___params___ - contains global variables like the localization of the dataset and the trained models, and also global classes
    * ___tactode_pipeline___ - image processing pipeline
    * ___testbench___ - test script
