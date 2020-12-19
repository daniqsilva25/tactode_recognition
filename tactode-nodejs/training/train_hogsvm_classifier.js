const cv = require('opencv4nodejs')
const fs = require('fs')
const path = require('path')
const params = require('../params.js')

// FUNCTIONS
function cutPiece (imgPath = '') {
  const img = cv.imread(imgPath)
  const gray = img.cvtColor(cv.COLOR_BGR2GRAY, 0)
  const bw = gray.threshold(127, 255, cv.THRESH_BINARY)
  const cnt = bw.findContours(cv.RETR_LIST, cv.CHAIN_APPROX_NONE, new cv.Point2(0, 0))
    .sort((c0, c1) => c1.area - c0.area)
  const boundSquare = cnt[1].boundingRect().toSquare()
  return img.getRegion(boundSquare)
}

function computeDescriptors (square = new cv.Mat()) {
  const roi = square.resize(params.classification.size, params.classification.size, 0, 0, cv.INTER_AREA)
  cv.imshow('roi', roi.rescale(5))
  cv.waitKey(1)
  const desc = hog.compute(roi)
  return desc
}

// MAIN PROGRAM
console.log(`train classifier started: ${params.getCurrentTime()}`)

const hog = new cv.HOGDescriptor({
  winSize: new cv.Size(params.classification.size, params.classification.size),
  blockSize: new cv.Size(16, 16),
  cellSize: new cv.Size(8, 8),
  blockStride: new cv.Size(8, 8),
  nbins: 9,
  gammaCorrection: true,
  signedGradient: true
})

const svm = new cv.SVM({
  kernelType: cv.ml.SVM.LINEAR,
  c: 2.5
})

const samples = []
const labels = []
let label = 0; let typeCounter = 0; let pieceCounter = 0
fs.readdirSync(params.classification.train).forEach(type => {
  console.debug(`\n > loading '${type}' pieces ...`)
  const typeFolder = path.join(params.classification.train, type)
  fs.readdirSync(typeFolder).forEach(piece => {
    console.debug(`   * computing descriptors for '${piece}' ...`)
    const pieceFolder = path.join(typeFolder, piece)
    label = 100 * typeCounter + pieceCounter
    fs.readdirSync(pieceFolder).forEach(img => {
      const imgPath = path.join(pieceFolder, img)
      const descriptors = computeDescriptors(cutPiece(imgPath))
      samples.push(descriptors)
      labels.push(label)
    })
    pieceCounter++
  })
  pieceCounter = 0
  typeCounter++
}); cv.destroyAllWindows()

console.debug('\n > shuffling data ...')
// Fisher-Yates shuffle algorithm
for (let i = labels.length - 1; i > 0; --i) {
  const j = Math.floor(Math.random() * (i + 1));
  [labels[i], labels[j]] = [labels[j], labels[i]];
  [samples[i], samples[j]] = [samples[j], samples[i]]
}

const trainData = new cv.TrainData(
  new cv.Mat(samples, cv.CV_32F),
  cv.ml.ROW_SAMPLE,
  new cv.Mat([labels], cv.CV_32S)
)

console.debug('\n > training ...')
svm.trainAuto(trainData, 10)
svm.save(params.classification.svm)
hog.save(params.classification.hog)

console.log(`\ntrain classifier ended:   ${params.getCurrentTime()}`)
