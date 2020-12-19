## JavaScript version of the developed software

### Folder structure and scripts/files explanation
* **testing/**
    * __config_tests_js.txt__ - recognition methods to test (change this file according to the methods you want to use)
    * __pieces.txt__ - tile names (__DO NOT CHANGE THIS FILE!__)
* **tools/**
    * ___frame_extractor.js___ - script for extracting frames from videos
    * ___make_templates.js___ - script for making templates from the dataset (forming the content of the "_dataset/templates/_" folder)
* **training/** - folder containing _.js_ scripts for training
* **remaining \*.js scripts**
    * ___classification___ - contains a function for each method to perform tile classification
    * ___params___ - contains global variables like the localization of the dataset and the trained models, and also global classes
    * ___tactode_pipeline___ - image processing pipeline
    * ___testbench___ - test script
