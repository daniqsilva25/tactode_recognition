## JavaScript version of the developed software

### Folder structure and scripts/files explanation
* **testing/**
    * __config_tests_js.txt__ - recognition methods to test (change this file according to the methods you want to use)
    * __pieces.txt__ - tile names (__DO NOT CHANGE THIS FILE!__)
* **training/** - folder containing JS scripts for training
* **remaining JS scripts**
    * __classification.js__ - contains a function for each method to perform tile classification
    * __params.js__ - contains global variables like the localization of the dataset and the trained models, and also global classes
    * __tactode_pipeline.js__ - image processing pipeline
    * __testbench.js__ - test script
