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
  let boundRect = cnt[1].boundingRect()
  const width = boundRect.width
  let height = Math.round(boundRect.height / 3)
  const x = boundRect.x
  const y = boundRect.y + 2 * height
  height = Math.round(width / 2)
  boundRect = new cv.Rect(x, y, width, height)
  return img.getRegion(boundRect)
}

function computeDescriptors (square = new cv.Mat()) {
  const roi = square.resize(params.detection.size.height, params.detection.size.width, 0, 0, cv.INTER_AREA)
  cv.imshow('roi', roi.rescale(2))
  cv.waitKey(1)
  const desc = hog.compute(roi)
  return desc
}

// MAIN PROGRAM
console.log(`train detector started: ${params.getCurrentTime()}`)

const hog = new cv.HOGDescriptor({
  winSize: new cv.Size(params.detection.size.width, params.detection.size.height),
  blockSize: new cv.Size(16, 16),
  cellSize: new cv.Size(8, 8),
  blockStride: new cv.Size(8, 8),
  nbins: 9,
  gammaCorrection: true,
  signedGradient: true
})

const svm = new cv.SVM({
  kernelType: cv.ml.SVM.LINEAR,
  c: 100
})

const samples = []
const labels = []
fs.readdirSync(params.detection.train).forEach(type => {
  console.debug(`\n > loading '${type}' samples ...`)
  const typeFolder = path.join(params.detection.train, type)
  const label = (type === 'positive') ? 1 : -1
  fs.readdirSync(typeFolder).forEach(piece => {
    console.debug(`   * computing descriptors for '${piece}' ...`)
    const pieceFolder = path.join(typeFolder, piece)
    fs.readdirSync(pieceFolder).forEach(img => {
      const imgPath = path.join(pieceFolder, img)
      const descriptors = computeDescriptors(cutPiece(imgPath))
      samples.push(descriptors)
      labels.push(label)
    })
  })
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
svm.save(params.detection.svm)
hog.save(params.detection.hog)

console.log(`\ntrain detector ended:   ${params.getCurrentTime()}`)
