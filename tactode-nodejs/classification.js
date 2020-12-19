const fs = require('fs')
const path = require('path')
const cv = require('opencv4nodejs')
const params = require('./params')

// FUNCTIONS
function runHogSvm (svm = new cv.SVM(), hog = new cv.HOGDescriptor(), config = new params.HogSvmClassifier(), res = new params.PipelineResult()) {
  let tLine = []
  const tilesArr = []
  let numTiles = 0
  res.rectsArr.forEach(line => {
    line.forEach(tr => {
      const roi = res.srcImg.getRegion(tr.rect)
      const resized = roi.resize(config.size.height, config.size.width, 0, 0, cv.INTER_AREA)
      const descriptors = hog.compute(resized)
      const prediction = svm.predict(descriptors)
      const typeIndex = Math.floor(prediction / 100)
      const pieceIndex = prediction % 100
      const classLine = config.classes[typeIndex].split(':')
      const tile = classLine[1].split(',')[pieceIndex]
      tLine.push(tile)
      numTiles++
    })
    tilesArr.push(tLine)
    tLine = []
  })
  return new params.ClassificationResult(tilesArr, numTiles, res.isRotated)
}

function runTemplateMatching (tmType = '', config = new params.TemplateMatchFeatDetDesc(), res = new params.PipelineResult()) {
  const templates = fs.readdirSync(params.DATASET.templates)
  let tLine = []
  const tilesArr = []
  let matchesArr = []
  let numTiles = 0
  res.rectsArr.forEach(line => {
    line.forEach(tr => {
      const roi = res.srcImg.getRegion(tr.rect)
      const resized = roi.resize(config.size.height, config.size.height, 0, 0, cv.INTER_AREA)
      templates.forEach(tmpl => {
        const pathTmpl = path.join(params.DATASET.templates, tmpl)
        const template = cv.imread(pathTmpl)
        const mask = new cv.Mat()
        let dst = new cv.Mat()
        if (tmType === 'TM_CCOEFF') {
          dst = resized.matchTemplate(template, cv.TM_CCOEFF, mask)
        } else if (tmType === 'TM_SQDIFF') {
          dst = resized.matchTemplate(template, cv.TM_SQDIFF, mask)
        }
        const minMax = dst.minMaxLoc(mask)
        matchesArr.push({ piece: tmpl.split('-')[0], res: minMax })
      })
      if (tmType === 'TM_CCOEFF') {
        matchesArr.sort((m0, m1) => m1.res.maxVal - m0.res.maxVal)
      } else if (tmType === 'TM_SQDIFF') {
        matchesArr.sort((m0, m1) => m0.res.minVal - m1.res.minVal)
      }
      tLine.push(matchesArr[0].piece)
      numTiles++
      matchesArr = []
    })
    tilesArr.push(tLine)
    tLine = []
  })
  return new params.ClassificationResult(tilesArr, numTiles, res.isRotated)
}

function runFeaturesDetDescMatch (fdd, bf, config = new params.TemplateMatchFeatDetDesc(), res = new params.PipelineResult()) {
  const templates = fs.readdirSync(params.DATASET.templates)
  let tLine = []
  const tilesArr = []
  let matchesArr = []
  let numTiles = 0
  res.rectsArr.forEach(line => {
    line.forEach(tr => {
      const roi = res.srcImg.getRegion(tr.rect)
      const resized = roi.resize(config.size.height, config.size.width, 0, 0, cv.INTER_CUBIC)
      const pieceKps = fdd.detect(resized)
      const pieceDpts = fdd.compute(resized, pieceKps)
      templates.forEach(tmpl => {
        const pathTmpl = path.join(params.DATASET.templates, tmpl)
        const template = cv.imread(pathTmpl)
        const templateKps = fdd.detect(template)
        const templateDpts = fdd.compute(template, templateKps)
        const knnMatches = bf.knnMatch(pieceDpts, templateDpts, 1)
        if (knnMatches.length > 0) {
          const piecePointArr = []; const templatePointArr = []
          const goodMatches = []
          for (let i = 0; i < knnMatches.length; ++i) {
            if (knnMatches[i].length > 0) {
              piecePointArr.push(pieceKps[knnMatches[i][0].queryIdx].pt)
              templatePointArr.push(templateKps[knnMatches[i][0].trainIdx].pt)
              goodMatches.push(knnMatches[i][0])
            }
          }
          const H = cv.findHomography(piecePointArr, templatePointArr, cv.RANSAC, 5)
          const bestMatches = []
          for (let i = 0; i < H.mask.rows; ++i) {
            if (H.mask.at(i, 0) === 1) {
              bestMatches.push(goodMatches[i])
            }
          }
          matchesArr.push({ piece: tmpl.split('-')[0], matches: bestMatches })
        }
      })
      matchesArr.sort((m0, m1) => m1.matches.length - m0.matches.length)
      tLine.push(matchesArr[0].piece)
      numTiles++
      matchesArr = []
    })
    tilesArr.push(tLine)
    tLine = []
  })
  return new params.ClassificationResult(tilesArr, numTiles, res.isRotated)
}

module.exports = {
  runHogSvm, runTemplateMatching, runFeaturesDetDescMatch
}
