## JavaScript version of the developed software

### Folder structure and scripts/files explanation
* **testing/**
    * __config_tests_js.txt__ - recognition methods to test (change this file according to the methods you want to use)
    * __pieces.txt__ - tile names (__DO NOT CHANGE THIS FILE!__)
* **training/** - folder containing _.js_ scripts for training
* **remaining \*.js scripts**
    * ___classification___ - contains a function for each method to perform tile classification
    * ___params___ - contains global variables like the localization of the dataset and the trained models, and also global classes
    * ___tactode_pipeline___ - image processing pipeline
    * ___testbench___ - test script
