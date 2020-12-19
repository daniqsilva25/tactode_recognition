const fs = require('fs')

// GLOBAL VARIABLES
const TESTS = {
  config: 'testing/config_tests_js.txt',
  tiles: 'testing/pieces.txt'
}

const RESULTS = {
  folder: '../../results/',
  file: 'results.txt'
}

const DATASET = {
  templates: '../dataset/templates',
  test_small: '../dataset/test/tactode_small',
  test_big: '../dataset/test/tactode_big',
  train_cla: '../dataset/train/tile_classification',
  train_det: '../dataset/train/teeth_detection'
}

// CLASSES
class PipelineResult {
  constructor (rects, src, isRotated) {
    this.rectsArr = rects
    this.srcImg = src
    this.isRotated = isRotated
  }
}

class ClassificationResult {
  constructor (tiles = [], numTiles = -1, isRotated = false) {
    this.tilesArr = tiles
    this.numberOfTiles = numTiles
    this.isRotated = isRotated
  }
}

class Size {
  constructor (w = -1, h = -1) {
    this.width = w
    this.height = h
  }
}

class HogSvmDetector {
  constructor () {
    this.svmFile = '../trained_models/hogsvm_JS/svm_detector_JS.yaml'
    this.hogFile = '../trained_models/hogsvm_JS/hog_detector.yaml'
    this.size = new Size(64, 32)
  }
}

class HogSvmClassifier {
  constructor () {
    this.classesFile = '../trained_models/hogsvm_JS/pieces.txt'
    this.svmFile = '../trained_models/hogsvm_JS/svm_classifier_JS.yaml'
    this.hogFile = '../trained_models/hogsvm_JS/hog_classifier.yaml'
    this.size = new Size(32, 32)
    this.classes = []
  }

  loadClasses () {
    this.classes = fs.readFileSync(this.classesFile).toString().split('\n')
  }
}

class TemplateMatchFeatDetDesc {
  constructor () {
    this.folder = DATASET.templates
    this.size = new Size(350, 350)
  }
}

// FUNCTIONS
function getCurrentTime () {
  const date = new Date()
  let dayOfMonth = date.getDate(); dayOfMonth = (dayOfMonth < 10) ? `0${dayOfMonth}` : dayOfMonth
  let month = date.getMonth(); month = (month < 10) ? `0${month + 1}` : month + 1
  let year = date.getFullYear(); year = (year < 10) ? `0${year}` : year
  let hours = date.getHours(); hours = (hours < 10) ? `0${hours}` : hours
  let minutes = date.getMinutes(); minutes = (minutes < 10) ? `0${minutes}` : minutes
  let seconds = date.getSeconds(); seconds = (seconds < 10) ? `0${seconds}` : seconds
  return `${dayOfMonth}/${month}/${year} ${hours}:${minutes}:${seconds}`
}

module.exports = {
  TESTS,
  RESULTS,
  DATASET,
  PipelineResult,
  ClassificationResult,
  Size,
  HogSvmDetector,
  HogSvmClassifier,
  TemplateMatchFeatDetDesc,
  getCurrentTime
}
