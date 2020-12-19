const fs = require('fs')
const path = require('path')
const cv = require('opencv4nodejs')
const params = require('./params.js')
const classification = require('./classification.js')
const pipeline = require('./tactode_pipeline.js')

// FUNCTIONS
function setupSaveDir (filePath = '', folderPath = '') {
  if (fs.existsSync(folderPath)) {
    fs.readdirSync(folderPath).forEach(file => {
      fs.unlinkSync(path.join(folderPath, file))
    })
  } else {
    fs.mkdirSync(folderPath)
  }
  if (filePath.length > 0) {
    fs.openSync(filePath, 'w')
  }
}

function initConfusions (filePath = '', tactodePiecesFile = '') {
  const confusions = []
  const auxConfMatrix = []
  fs.readFileSync(tactodePiecesFile, 'utf-8').split('\n').forEach(line => {
    if (line.length > 2) {
      const pieces = line.split(':')[1].split(',')
      pieces.forEach(p => {
        confusions.push({ id: p, tp: 0, fp: 0, fn: 0, confMatrix: [] })
        auxConfMatrix.push(p)
      })
    }
  })
  for (let i = 0; i < confusions.length; ++i) {
    for (let j = 0; j < auxConfMatrix.length; ++j) {
      confusions[i].confMatrix.push({ id: auxConfMatrix[j], count: 0 })
    }
  }
  fs.openSync(filePath, 'w')
  return confusions
}

function updateConfusions (confusions = [], expectedPiece = '', obtainedPiece = '') {
  confusions.forEach(p => {
    if (p.id === obtainedPiece) {
      p.confMatrix.forEach(cm => {
        if (cm.id === expectedPiece) {
          cm.count++
        }
      })
    }
    if (expectedPiece === obtainedPiece) {
      if (p.id === obtainedPiece) {
        p.tp++
      }
    } else {
      if (p.id === obtainedPiece) {
        p.fp++
      } else if (p.id === expectedPiece) {
        p.fn++
      }
    }
  })
}

function getExpectedResult (fileName = '', folderPath = '') {
  const tArr = []
  let numPieces = 0
  const codeTxtFile = `${path.join(folderPath, fileName)}.txt`
  if (fs.existsSync(codeTxtFile)) {
    let fileContents = fs.readFileSync(codeTxtFile, 'utf-8').split('\n')
    fileContents = (fileContents[fileContents.length - 1] === '') ? fileContents.splice(0, fileContents.length - 1) : fileContents
    fileContents.forEach(cont => {
      const line = cont.split(',')
      numPieces += line.length
      tArr.push(line)
    })
  } else {
    console.error(`ERROR: The file '${codeTxtFile}' does not exist!`)
  }
  return new params.ClassificationResult(tArr, numPieces, false)
}

function getPiecesEvaluation (confusions = [], obtained = new params.ClassificationResult(), expected = new params.ClassificationResult()) {
  let numErrors = 0
  const errorsArr = []
  if (obtained.isRotated === true) {
    numErrors = -2
  } else if (obtained.numberOfTiles === expected.numberOfTiles) {
    for (let i = 0; i < obtained.tilesArr.length; i++) {
      for (let j = 0; j < obtained.tilesArr[i].length; j++) {
        const obtainedPiece = obtained.tilesArr[i][j]
        const expectedPiece = expected.tilesArr[i][j]
        if (obtainedPiece !== expectedPiece) {
          numErrors++
          errorsArr.push([i, j, expectedPiece, obtainedPiece])
        }
        updateConfusions(confusions, expectedPiece, obtainedPiece)
      }
    }
  } else {
    numErrors = -1
  }
  return { errorsArr, numErrors }
}

function writeEvaluation (resFile = '', img = '', evaluation = { errorsArr: [], numErrors: -1 }) {
  let fileContent = `${img}\n`
  if (evaluation.numErrors === 0) {
    fileContent += '  [YES] -> Finished with success!\n'
  } else if (evaluation.numErrors === -1) {
    fileContent += '  [NO_1] -> Number of pieces does not match!\n'
  } else if (evaluation.numErrors === -2) {
    fileContent += '  [NO_2] -> The code is rotated!\n'
  } else {
    fileContent += '  [NO_3] -> Some pieces are mismatched!\n'
    evaluation.errorsArr.forEach(line => {
      fileContent += `    (${line[0]},${line[1]}): Expected = '${line[2]}' ... Obtained = '${line[3]}'\n`
    })
  }
  fs.appendFileSync(resFile, `${fileContent}\n\n`)
}

function writeConfusionsFile (confusionsFile = '', confusions = []) {
  let fileContent = ''
  let strConfMatrix = ''
  confusions.forEach(c => {
    strConfMatrix = ''
    c.confMatrix.forEach(cm => {
      strConfMatrix += `'${cm.id}':${cm.count}`
      if (c.confMatrix.indexOf(cm) < c.confMatrix.length - 1) {
        strConfMatrix += ' , '
      }
    })
    fileContent += `'${c.id}' -> { tp:${c.tp} , fp:${c.fp} , fn:${c.fn} , { ${strConfMatrix} } }\n\n`
  })
  fs.appendFileSync(confusionsFile, fileContent)
}

function writeConfusionMatrix (confMatrixFile = '', confusions = []) {
  let contentStr = ''
  confusions.forEach(elem => {
    elem.confMatrix.forEach(cElem => {
      contentStr += `${cElem.count}`
      if (elem.confMatrix.indexOf(cElem) !== elem.confMatrix.length - 1) {
        contentStr += ','
      }
    })
    contentStr += '\n'
  })
  fs.appendFileSync(confMatrixFile, contentStr)
}

function writeLatexTableOfConfusions (latexTableFile = '', confusions = []) {
  let contentStr = ''
  confusions.forEach(elem => {
    let tileName = elem.id
    if (tileName.includes('_')) {
      const twoParts = tileName.split('_')
      tileName = `${twoParts[0]} ${twoParts[1]}`
    }
    contentStr += `${tileName} `
    const tp = elem.tp
    const fp = elem.fp
    const fn = elem.fn
    contentStr += `& ${tp} & ${fp} & ${fn} `
    const precision = (tp === 0 && fp === 0) ? '-' : Math.floor(tp * 1000 / (tp + fp)) / 1000
    const recall = (tp === 0 && fn === 0) ? '-' : Math.floor(tp * 1000 / (tp + fn)) / 1000
    const f1Score = ((precision === 0 && recall === 0) || precision === '-' || recall === '-') ? '-' : Math.floor(2 * precision * recall * 1000 / (precision + recall)) / 1000
    if (precision !== '-') {
      contentStr += `& ${precision.toPrecision(3)} `
    } else {
      contentStr += `& ${precision} `
    }
    if (recall !== '-') {
      contentStr += `& ${recall.toPrecision(3)} `
    } else {
      contentStr += `& ${recall} `
    }
    if (f1Score !== '-') {
      contentStr += `& ${f1Score.toPrecision(3)} `
    } else {
      contentStr += `& ${f1Score} `
    }
    contentStr += '\\\\ \\hline \n'
  })
  fs.appendFileSync(latexTableFile, contentStr)
}

// MAIN PROGRAM
console.log(`test started: ${params.getCurrentTime()}\n`)

const tMfDDM = new params.TemplateMatchFeatDetDesc()
const hogSvmDetection = new params.HogSvmDetector()
const hogDetector = new cv.HOGDescriptor()
hogDetector.load(hogSvmDetection.hogFile)
const svmDetector = new cv.SVM()
svmDetector.load(hogSvmDetection.svmFile)

const testQuantity = process.argv[2]
if (testQuantity === 'all') {
  let classificationDataset = ''
  let mainResultsFolder = ''
  if (process.argv.length > 3) {
    if (process.argv[3] === 'small') {
      classificationDataset = params.DATASET.test_small
      mainResultsFolder = params.RESULTS.folder + 'small'
    } else if (process.argv[3] === 'big') {
      classificationDataset = params.DATASET.test_big
      mainResultsFolder = params.RESULTS.folder + 'big'
    } else {
      console.error('\nERROR: You must specify a VALID test dataset (small/big) to use!\n')
      process.exit(-1)
    }
  } else {
    console.error('\nERROR: You must specify the test dataset (small/big) to use!\n')
    process.exit(-1)
  }
  const resultsFile = mainResultsFolder + '/' + params.RESULTS.file
  fs.openSync(resultsFile, 'w')
  let resFileContent = ''
  resFileContent += 'Caption:\n'
  resFileContent += '.NoE3  = Number of misclassified pieces\n'
  resFileContent += '.NoI   = Number of Images tested\n'
  resFileContent += '.NoP   = Number of Pieces tested\n'
  resFileContent += '.Acc   = Accuracy\n'
  resFileContent += '.AET   = Average Execution Time\n'
  resFileContent += '.SDET  = Standard Deviation Execution Time\n'
  resFileContent += '.minET = Minimum Execution Time\n'
  resFileContent += '.maxET = Maximum Execution Time\n\n'
  fs.appendFileSync(resultsFile, resFileContent)
  const methodsArr = fs.readFileSync(params.TESTS.config, 'utf-8').split('\n').filter(m => m.length > 0)
  methodsArr.forEach(classifMethod => {
    resFileContent = ''
    let hogSvmClassification = new params.HogSvmClassifier()
    let hogClassifier = new cv.HOGDescriptor()
    let svmClassifier = new cv.SVM()
    let bf = new cv.BFMatcher(cv.NORM_HAMMING, true)
    let fdd = new cv.ORBDetector(600)
    if (classifMethod === 'HOG' || classifMethod === 'ORB' || classifMethod === 'SURF' || classifMethod === 'SIFT' || classifMethod === 'BRISK' || classifMethod === 'TM_CCOEFF' || classifMethod === 'TM_SQDIFF') {
      console.log(`\n\n <------- ${classifMethod} | ${params.getCurrentTime()} ------->\n`)

      if (classifMethod === 'HOG') {
        hogSvmClassification = new params.HogSvmClassifier()
        hogSvmClassification.loadClasses()
        hogClassifier = new cv.HOGDescriptor()
        hogClassifier.load(hogSvmClassification.hogFile)
        svmClassifier = new cv.SVM()
        svmClassifier.load(hogSvmClassification.svmFile)
      } else if (classifMethod === 'ORB' || classifMethod === 'BRISK' || classifMethod === 'SURF' || classifMethod === 'SIFT') {
        bf = new cv.BFMatcher(cv.NORM_HAMMING, true)
        if (classifMethod === 'ORB') {
          fdd = new cv.ORBDetector(600)
        } else if (classifMethod === 'BRISK') {
          fdd = new cv.BRISKDetector(10)
        } else {
          bf = new cv.BFMatcher(cv.NORM_L1, true)
          if (classifMethod === 'SURF') {
            fdd = new cv.SURFDetector(50)
          } else if (classifMethod === 'SIFT') {
            fdd = new cv.SIFTDetector(600)
          }
        }
      }

      const methodResultFolder = path.join(mainResultsFolder, classifMethod)
      if (!fs.existsSync(methodResultFolder)) {
        fs.mkdirSync(methodResultFolder)
      }

      const confusionsFile = path.join(methodResultFolder, 'confusions.txt')
      const confusions = initConfusions(confusionsFile, params.TESTS.tiles)

      resFileContent += `<--- ${classifMethod} --->\n`
      fs.appendFileSync(resultsFile, resFileContent)

      const execTimesArr = []
      let sumExecTimes = 0
      let numberOfPics = 0
      let numberOfPieces = 0
      let globalNumberOfPieces = 0
      let numTotalErrors = 0
      let globalNumTotalErrors = 0
      let numCodesWithErrors = 0
      let globalNumCodesWithErrors = 0
      let numErrorsType1 = 0
      let globalNumErrorsType1 = 0
      let numErrorsType2 = 0
      let globalNumErrorsType2 = 0
      let numErrorsType3 = 0
      let globalNumErrorsType3 = 0

      fs.readdirSync(classificationDataset).forEach(resolution => {
        resFileContent = ''
        resFileContent += `  * ${resolution} -> `
        numberOfPieces = 0
        numTotalErrors = 0
        numCodesWithErrors = 0
        numErrorsType1 = 0
        numErrorsType2 = 0
        numErrorsType3 = 0
        console.log(`    * testing '${resolution}' ...`)

        const resolutionTestFolder = path.join(classificationDataset, resolution)
        const resolutionResultFolder = path.join(methodResultFolder, resolution)
        if (!fs.existsSync(resolutionResultFolder)) {
          fs.mkdirSync(resolutionResultFolder)
        }

        fs.readdirSync(resolutionTestFolder).forEach(folder => {
          const codesResultFolder = path.join(resolutionResultFolder, folder)
          const evalCodesFile = path.join(codesResultFolder, `aval-${folder}.txt`)
          setupSaveDir(evalCodesFile, codesResultFolder)

          const codesTestFolder = path.join(resolutionTestFolder, folder)
          const expectedTiles = getExpectedResult(folder, codesTestFolder)
          if (expectedTiles.numberOfTiles > 0) {
            fs.readdirSync(codesTestFolder).forEach(img => {
              if (img.split('.')[1] === 'jpg') {
                const imgPath = path.join(codesTestFolder, img)
                const startTime = process.hrtime()

                const pipeRes = pipeline.runMain(imgPath, hogDetector, svmDetector, hogSvmDetection)
                let obtainedTiles = new params.ClassificationResult()
                if (classifMethod === 'HOG') {
                  obtainedTiles = classification.runHogSvm(
                    svmClassifier, hogClassifier, hogSvmClassification, pipeRes
                  )
                } else if (classifMethod === 'ORB' || classifMethod === 'BRISK' || classifMethod === 'SIFT' || classifMethod === 'SURF') {
                  obtainedTiles = classification.runFeaturesDetDescMatch(
                    fdd, bf, tMfDDM, pipeRes
                  )
                } else if (classifMethod === 'TM_CCOEFF' || classifMethod === 'TM_SQDIFF') {
                  obtainedTiles = classification.runTemplateMatching(
                    classifMethod, tMfDDM, pipeRes
                  )
                }

                const endTime = process.hrtime(startTime)
                const execTime = endTime[0] + endTime[1] / 1000000000
                execTimesArr.push(execTime)
                sumExecTimes += execTime
                numberOfPics++
                numberOfPieces += expectedTiles.numberOfTiles
                // const aux = {resArr: [['repeat', 'lower_a'], ['forward'], ['right'], ['end_repeat']], numPieces: 5};
                const evaluation = getPiecesEvaluation(confusions, obtainedTiles, expectedTiles)
                if (evaluation.numErrors !== 0) {
                  numCodesWithErrors++
                  if (evaluation.numErrors > 0) {
                    numErrorsType3 += evaluation.numErrors
                  } else if (evaluation.numErrors === -1) {
                    numErrorsType1++
                  } else if (evaluation.numErrors === -2) {
                    numErrorsType2++
                  }
                }
                numTotalErrors = numErrorsType1 + numErrorsType2 + numErrorsType3
                writeEvaluation(evalCodesFile, img, evaluation)
              }
            })
          } else {
            console.error(`ERROR: Missing content on file '${folder}.txt'!`)
          }
        })
        console.log('      > Number of codes with errors:', numCodesWithErrors)
        console.log('      > Number of errors of type 1:', numErrorsType1)
        console.log('      > Number of errors of type 2:', numErrorsType2)
        console.log('      > Number of errors of type 3:', numErrorsType3)
        console.log('      > Total amount of errors:', numTotalErrors)
        console.log('      > Number of pieces tested:', numberOfPieces)
        const acc = Math.round((numberOfPieces - numErrorsType3) * 10000 / numberOfPieces) / 100
        console.log('      > Accuracy (%):', acc)
        console.log()
        resFileContent += ` NoE3: ${numErrorsType3} | NoP: ${numberOfPieces} | Acc: ${acc} %\n`
        fs.appendFileSync(resultsFile, resFileContent)
        globalNumCodesWithErrors += numCodesWithErrors
        globalNumErrorsType1 += numErrorsType1
        globalNumErrorsType2 += numErrorsType2
        globalNumErrorsType3 += numErrorsType3
        globalNumTotalErrors += numTotalErrors
        globalNumberOfPieces += numberOfPieces
      })
      writeConfusionsFile(confusionsFile, confusions)
      const latexTableFile = path.join(methodResultFolder, `latex-${classifMethod}.txt`)
      fs.openSync(latexTableFile, 'w')
      writeLatexTableOfConfusions(latexTableFile, confusions)
      const confMatrixFile = path.join(methodResultFolder, `confMatrix-${classifMethod}.csv`)
      fs.openSync(confMatrixFile, 'w')
      writeConfusionMatrix(confMatrixFile, confusions)
      console.log('  +++++ Global -> %s +++++', classifMethod)
      console.log('   Number of codes with errors:', globalNumCodesWithErrors)
      console.log('   Number of errors of type 1:', globalNumErrorsType1)
      console.log('   Number of errors of type 2:', globalNumErrorsType2)
      console.log('   Number of errors of type 3:', globalNumErrorsType3)
      console.log('   Total amount of errors:', globalNumTotalErrors)
      console.log('\n   Number of pictures tested:', numberOfPics)
      console.log('   Number of pieces tested:', globalNumberOfPieces)
      const globalAcc = Math.round((globalNumberOfPieces - globalNumErrorsType3) * 10000 / globalNumberOfPieces) / 100
      const avgExecTime = Math.round(sumExecTimes * 1000 / numberOfPics) / 1000
      let sumMean = 0
      execTimesArr.forEach(currExecTime => {
        sumMean += Math.pow(currExecTime - avgExecTime, 2)
      })
      const stdDeviationExecTime = Math.round(Math.sqrt(sumMean / numberOfPics) * 1000) / 1000
      const maxExecTime = Math.round(Math.max(...execTimesArr) * 1000) / 1000
      const minExecTime = Math.round(Math.min(...execTimesArr) * 1000) / 1000
      console.log('   Accuracy (%):', globalAcc)
      console.log('   Average execution time (seconds):', avgExecTime)
      console.log('   Standard deviation execution time (seconds):', stdDeviationExecTime)
      console.log('   Minimum execution time (seconds):', minExecTime)
      console.log('   Maximum execution time (seconds):', maxExecTime)
      console.log(`\n !------- ${classifMethod} | ${params.getCurrentTime()} -------!`)
      resFileContent = ''
      resFileContent += `  GLOBAL ${classifMethod}\n`
      resFileContent += `   - NoI: ${numberOfPics}\n`
      resFileContent += `   - NoP: ${globalNumberOfPieces}\n`
      resFileContent += `   - NoE3: ${globalNumErrorsType3}\n`
      resFileContent += `   - Acc: ${globalAcc} %\n`
      resFileContent += `   - AET: ${avgExecTime} seconds\n`
      resFileContent += `   - SDET: ${stdDeviationExecTime} seconds\n`
      resFileContent += `   - minET: ${minExecTime} seconds\n`
      resFileContent += `   - maxET: ${maxExecTime} seconds\n\n`
      fs.appendFileSync(resultsFile, resFileContent)
    } else {
      console.error('\nERROR: Unrecognized classification method! ->', classifMethod)
    }
  })
} else if (testQuantity === 'one') {
  if (process.argv.length !== 5) {
    console.error('\nERROR: You must specify the classification method (HOG/ORB/BRISK/SURF/SIFT/TM_CCOEFF/TM_SQDIFF) and the image path!\n')
    process.exit(-1)
  } else {
    const classifMethod = process.argv[3]
    const testImgPath = process.argv[4]
    console.time('main')
    const pipeRes = pipeline.runMain(testImgPath, hogDetector, svmDetector, hogSvmDetection)
    let isValid = false
    let obtainedTiles = new params.ClassificationResult()
    if (classifMethod === 'HOG') {
      isValid = true
      const hogSvmClassification = new params.HogSvmClassifier()
      hogSvmClassification.loadClasses()
      const hogClassifier = new cv.HOGDescriptor()
      hogClassifier.load(hogSvmClassification.hogFile)
      const svmClassifier = new cv.SVM()
      svmClassifier.load(hogSvmClassification.svmFile)
      obtainedTiles = classification.runHogSvm(
        svmClassifier, hogClassifier, hogSvmClassification, pipeRes
      )
    } else if (classifMethod === 'ORB' || classifMethod === 'BRISK' || classifMethod === 'SIFT' || classifMethod === 'SURF') {
      isValid = true
      let bf = new cv.BFMatcher(cv.NORM_HAMMING, true)
      let fdd = new cv.ORBDetector(600)
      if (classifMethod === 'ORB') {
        fdd = new cv.ORBDetector(600)
      } else if (classifMethod === 'BRISK') {
        fdd = new cv.BRISKDetector(10)
      } else {
        bf = new cv.BFMatcher(cv.NORM_L1, true)
        if (classifMethod === 'SIFT') {
          fdd = new cv.SIFTDetector(600)
        } else if (classifMethod === 'SURF') {
          fdd = cv.SURFDetector(50)
        }
      }
      obtainedTiles = classification.runFeaturesDetDescMatch(
        fdd, bf, new params.TemplateMatchFeatDetDesc(), pipeRes
      )
    } else if (classifMethod === 'TM_CCOEFF' || classifMethod === 'TM_SQDIFF') {
      isValid = true
      obtainedTiles = classification.runTemplateMatching(
        classifMethod, new params.TemplateMatchFeatDetDesc(), pipeRes
      )
    }
    console.timeEnd('main')
    if (isValid) {
      console.log('\nPieces found:', obtainedTiles.numberOfTiles)
      console.log(obtainedTiles.tilesArr)
    } else {
      console.log('\nERROR: You must specify a valid classification method!\n')
      process.exit(-1)
    }
  }
} else {
  console.error("ERROR: Please insert a valid option! ('one' or 'all')")
}

console.log(`\ntest ended:   ${params.getCurrentTime()}`)
