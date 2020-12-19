const cv = require('opencv4nodejs')
const params = require('./params.js')

// FUNCTIONS
function showContours (cnts = [], mat = new cv.Mat(), fld = '', apdx = '') {
  cnts = cnts.map(c => c.getPoints())
  for (let i = 0; i < cnts.length; ++i) {
    const color = new cv.Vec3(Math.random() * 255, Math.random() * 255, Math.random() * 255)
    mat.drawContours(cnts, i, color, 2, cv.LINE_AA)
  }
}

function drawRects (arr = [], mat = new cv.Mat(), fld = '', apdx = '') {
  arr.forEach(elem => {
    if (elem.length >= 1) {
      elem.forEach(subElem => {
        const color = new cv.Vec3(Math.random() * 255, Math.random() * 255, Math.random() * 255)
        mat.drawRectangle(subElem.rect, color, 2)
      })
    } else {
      const color = new cv.Vec3(Math.random() * 255, Math.random() * 255, Math.random() * 255)
      mat.drawRectangle(elem.rect, color, 2)
    }
  })
}

function unionRect (rect1 = new cv.Rect(), rect2 = new cv.Rect()) {
  const top1 = rect1.y; const bottom1 = rect1.y + rect1.height
  const left1 = rect1.x; const right1 = rect1.x + rect1.width
  const top2 = rect2.y; const bottom2 = rect2.y + rect2.height
  const left2 = rect2.x; const right2 = rect2.x + rect2.width

  const top = (top1 <= top2) ? top1 : top2
  const bottom = (bottom1 >= bottom2) ? bottom1 : bottom2
  const left = (left1 <= left2) ? left1 : left2
  const right = (right1 >= right2) ? right1 : right2

  const width = right - left
  const height = bottom - top

  return new cv.Rect(left, top, width, height)
}

function toSquare (rect = new cv.Rect()) {
  const diff = Math.round(Math.abs(rect.width - rect.height) / 2)
  const side = (rect.width >= rect.height) ? rect.width : rect.height
  const x = (rect.width >= rect.height) ? rect.x : rect.x - diff
  const y = (rect.height >= rect.width) ? rect.y : rect.y - diff

  return new cv.Rect(x, y, side, side)
}

const newContour = (idx = -1, hierarchy = new cv.Vec4(-1, -1, -1, -1), rect = new cv.Rect(),
  center = new cv.Point2(-1, -1), area = -1, perimeter = -1,
  aspectRatio = -1, extent = -1, solidity = -1, equiDiameter = -1) => {
  return {
    idx: idx,
    haveTeeth: false,
    hierarchy: hierarchy,
    rect: rect,
    center: {
      x: center.x,
      y: center.y
    },
    properties: {
      area: area,
      perimeter: perimeter,
      aspectRatio: aspectRatio,
      extent: extent,
      solidity: solidity,
      equiDiameter: equiDiameter
    },
    updateCenter: function () {
      this.center.x = this.rect.x + Math.round(this.rect.width / 2)
      this.center.y = this.rect.y + Math.round(this.rect.height / 2)
    }
  }
}

const newPiece = (rect = new cv.Rect()) => {
  return {
    rect: rect,
    tactode: {
      type: '',
      piece: ''
    }
  }
}

function runMain (imgPath = '', hog = new cv.HOGDescriptor(), svm = new cv.SVM(), config = new params.HogSvmDetector()) {
  // -> Beginning
  let src = cv.imread(imgPath)

  // -> Resizing image
  const normSize = 1920
  let rows = 0; let cols = 0; let aspectRatio = 0
  if (src.rows !== normSize && src.cols !== normSize) {
    if (src.rows >= src.cols) {
      aspectRatio = src.rows / src.cols
      rows = normSize
      cols = Math.floor(normSize / aspectRatio)
    } else {
      aspectRatio = src.cols / src.rows
      cols = normSize
      rows = Math.floor(normSize / aspectRatio)
    }
    src = src.resize(rows, cols, 0, 0, cv.INTER_AREA)
  }

  // -> Adjusting image
  let adjusted = src.copy(src)
  adjusted = cv.gaussianBlur(adjusted, new cv.Size(9, 9), 0, 0, cv.BORDER_DEFAULT)
  adjusted = adjusted.resize(0, 0, 0.3, 0.3, cv.INTER_AREA)
  adjusted = adjusted.cvtColor(cv.COLOR_BGR2GRAY, 0).canny(100, 200, 3)
  const lines = adjusted.houghLines(1, Math.PI / 180, 50, 0, 0, 0, Math.PI)
  if (lines.length > 0) {
    let angle = -1
    const bigAngles = []
    for (let i = 0; i < lines.length; ++i) {
      const degrees = lines[i].y * 180 / Math.PI
      if (angle === -1) {
        angle = degrees
        bigAngles.push(degrees)
      } else {
        if (Math.abs(degrees - angle) > 40 && Math.abs(degrees - angle) < 140) {
          bigAngles.push(degrees)
          break
        }
      }
    }
    bigAngles.forEach(a => {
      let diffAngle = a
      const idx = bigAngles.indexOf(a)
      if (diffAngle > 45 && diffAngle <= 135) {
        diffAngle = diffAngle - 90
      } else if (diffAngle > 135) {
        diffAngle = diffAngle - 180
      } else {
        diffAngle += 0
      }
      bigAngles.splice(idx, 1, diffAngle)
    })
    let adjustingAngle = -1
    if (bigAngles.length > 1) {
      if (bigAngles[0] * bigAngles[1] < 0) {
        adjustingAngle = (bigAngles[0] - bigAngles[1]) / 2
      } else {
        adjustingAngle = (bigAngles[0] + bigAngles[1]) / 2
      }
    } else {
      adjustingAngle = bigAngles[0]
    }
    const rotCenter = new cv.Point2(src.cols / 2, src.rows / 2)
    const rotMatrix = cv.getRotationMatrix2D(rotCenter, adjustingAngle, 1)
    const dsize = new cv.Size(src.cols, src.rows)
    src = src.warpAffine(rotMatrix, dsize, cv.INTER_LINEAR, cv.BORDER_DEFAULT, new cv.Vec3(0, 0, 0))
  }

  // -> Binarising image
  const blur = cv.gaussianBlur(src, new cv.Size(11, 11), 0, 0, cv.BORDER_DEFAULT)
  const kernel = cv.getStructuringElement(cv.MORPH_CROSS, new cv.Size(6, 6))
  const grad = blur.morphologyEx(kernel, cv.MORPH_GRADIENT)
  const bw = grad.inRange(new cv.Vec3(0, 0, 0), new cv.Vec3(20, 20, 20))
  let fg = bw.morphologyEx(kernel, cv.MORPH_OPEN)
  fg = fg.morphologyEx(kernel, cv.MORPH_CLOSE)

  // -> Segmenting and rotating image
  let rotationStep = 0
  let isRotated = true
  let bigger = newContour()
  let smaller = newContour()
  let cntArr = []
  let contours = [new cv.Contour()]
  while (rotationStep < 4 && isRotated) {
    cntArr = []
    bigger = newContour()
    smaller = newContour()

    // -> Finding contours
    contours = fg.findContours(cv.RETR_TREE, cv.CHAIN_APPROX_NONE, new cv.Point2(0, 0))

    // -> Getting contours' properties
    for (let i = 0; i < contours.length; ++i) {
      const cnt = contours[i]
      const hier = cnt.hierarchy

      const rect = cnt.boundingRect()
      const hull = cnt.convexHull(false)

      const rectArea = rect.width * rect.height
      const center = new cv.Point2(rect.x + rect.width / 2, rect.y + rect.height / 2)

      const area = cnt.area
      const perimeter = cnt.arcLength(false)
      const aspectRatio = rect.width / rect.height
      const extent = area / rectArea
      const solidity = area / hull.area
      const equiDiameter = Math.sqrt(4 * area / Math.PI)

      cntArr.push(newContour(i, hier, rect, center, area, perimeter,
        aspectRatio, extent, solidity, equiDiameter))
    }

    // -> Finding grandparent, parent and child contours
    cntArr.sort((c0, c1) => c1.properties.area - c0.properties.area)
    let grandparentidx = -1
    let parentidx = -1
    for (let i = 0; i < cntArr.length; ++i) {
      const ch = cntArr[i].hierarchy
      const ci = cntArr[i].idx
      if (grandparentidx === -1) {
        if (ch.z === -1 && ch.y !== -1) {
          grandparentidx = ci
        }
      } else {
        if (ch.z === grandparentidx) {
          parentidx = ci
          break
        }
      }
    }
    cntArr = cntArr.filter(c => c.hierarchy.z === parentidx)

    // -> Finding biggest and smallest tile-alike contours
    for (let i = 0; i < cntArr.length; ++i) {
      const c = cntArr[i]
      if (bigger.idx === -1 && c.properties.aspectRatio >= 0.7) {
        bigger = c
      } else if (bigger.idx !== -1 && c.properties.area / bigger.properties.area < 0.35) {
        smaller = cntArr[i - 1]
        break
      }
    }

    // -> Detecting teeth in the tiles
    for (let i = cntArr.indexOf(bigger); i <= cntArr.indexOf(smaller); ++i) {
      const c = cntArr[i]
      const width = c.rect.width
      let height = Math.round(c.rect.height / 3)
      const x = c.rect.x
      const y = c.rect.y + 2 * height
      height = Math.round(width / 2)
      const roi = src.getRegion(new cv.Rect(x, y, width, height))
        .resize(config.size.height, config.size.width, 0, 0, cv.INTER_AREA)
      const descriptors = hog.compute(roi)
      const prediction = svm.predict(descriptors)
      c.haveTeeth = (prediction === 1)
    }

    // -> Filtering teeth detections and identifying rotation
    const teethArr = cntArr.filter(c => c.haveTeeth)
    if (teethArr.length > 0) {
      teethArr.sort((t0, t1) => t0.rect.x - t1.rect.x)
      const mostLeft = teethArr[0]
      if (!cntArr.some(c => c.rect.x < mostLeft.rect.x - mostLeft.rect.width / 2)) {
        let prevTeeth = newContour()
        let columnCount = 1
        for (let i = 0; i < teethArr.length; ++i) {
          const currTeeth = teethArr[i]
          if (prevTeeth.idx === -1) {
            prevTeeth = currTeeth
          } else {
            if (Math.abs(prevTeeth.rect.x - currTeeth.rect.x) <= prevTeeth.rect.width / 2) {
              columnCount++
              prevTeeth = currTeeth
            }
            if (columnCount >= 2) {
              isRotated = false
              break
            }
          }
        }
      }
    }
    if (isRotated) {
      fg = fg.rotate(cv.ROTATE_90_CLOCKWISE)
      src = src.rotate(cv.ROTATE_90_CLOCKWISE)
    }
    rotationStep++
  }

  // -> Removing inner and noisy contours
  for (let i = 0; i < cntArr.length; ++i) {
    const currCnt = cntArr[i]
    if (currCnt.idx !== -1) {
      const currRectArea = currCnt.rect.width * currCnt.rect.height
      const subArr = cntArr.slice(i + 1)
      for (let j = 0; j < subArr.length; ++j) {
        const subC = subArr[j]
        if (subC.idx !== -1) {
          const resOrRect = unionRect(currCnt.rect, subC.rect)
          const resOrRectArea = resOrRect.width * resOrRect.height
          const idx = i + j + 1
          if (resOrRectArea / currRectArea === 1 || subC.properties.aspectRatio < 0.25 || subC.properties.aspectRatio > 5 || subC.properties.area / bigger.properties.area < 0.01 || subC.properties.extent <= 0.25) {
            contours.splice(cntArr[idx].idx, 1, new cv.Contour())
            cntArr.splice(idx, 1, newContour())
          }
        }
      }
    }
  }
  cntArr = cntArr.filter(c => c.idx !== -1)

  // -> Agglomerating little bits of remaining contours
  cntArr.sort((c0, c1) => c0.properties.area - c1.properties.area)

  for (let i = 0; i < cntArr.length; ++i) {
    const currCnt = cntArr[i]
    const currRectArea = currCnt.rect.width * currCnt.rect.height
    const currCenter = currCnt.center
    const slicedArr = cntArr.slice(i + 1)
    let subsubArr = [newContour()]
    let idx = -1
    for (let j = 0; j < slicedArr.length; ++j) {
      const subCnt = slicedArr[j]
      const subRectArea = subCnt.rect.width * subCnt.rect.height
      if (currRectArea / subRectArea < 0.25) {
        subsubArr = slicedArr.slice(j)
        idx = j
        break
      }
    }
    if (idx !== -1) {
      contours.splice(cntArr[i].idx, 1, new cv.Contour())
      cntArr.splice(i, 1, newContour())
      subsubArr.sort((c0, c1) => {
        const c0x = Math.pow((c0.center.x - currCenter.x), 2)
        const c0y = Math.pow((c0.center.y - currCenter.y), 2)
        const c1x = Math.pow((c1.center.x - currCenter.x), 2)
        const c1y = Math.pow((c1.center.y - currCenter.y), 2)
        const leftSide = Math.sqrt(c0x + c0y)
        const rightSide = Math.sqrt(c1x + c1y)
        const dist = Math.round(leftSide - rightSide)
        return dist
      })
      let closerElem = newContour()
      let next = false
      for (let k = 0; k < subsubArr.length; ++k) {
        const leftLimit = subsubArr[k].rect.x
        const rightLimit = subsubArr[k].rect.x + subsubArr[k].rect.width
        const topLimit = subsubArr[k].rect.y
        const bottomLimit = subsubArr[k].rect.y + subsubArr[k].rect.height
        if (currCenter.x >= leftLimit && currCenter.x <= rightLimit) {
          if (currCenter.y < subsubArr[k].center.y || (currCenter.y <= bottomLimit && currCenter.y >= topLimit)) {
            closerElem = subsubArr[k]
            break
          } else if (next) {
            closerElem = subsubArr[k - 1]
            break
          } else {
            next = true
          }
        } else {
          if (currCenter.y <= bottomLimit && currCenter.y >= topLimit) {
            closerElem = subsubArr[k]
            break
          } else if (next) {
            closerElem = subsubArr[k - 1]
            break
          } else {
            next = true
          }
        }
      }
      const otheridx = cntArr.indexOf(closerElem)
      if (otheridx !== -1) {
        cntArr[otheridx].rect = unionRect(cntArr[otheridx].rect, currCnt.rect)
        cntArr[otheridx].updateCenter()
      }
    }
  }
  cntArr = cntArr.filter(c => c.idx !== -1)

  // -> Positioning
  cntArr.sort((c0, c1) => c0.rect.y - c1.rect.y)
  const rectsArr = []; let rLine = []
  let prevCnt
  cntArr.forEach(currCnt => {
    if (prevCnt === undefined) {
      prevCnt = currCnt
    } else {
      if (currCnt.rect.y > prevCnt.rect.y + prevCnt.rect.height / 4) { // new line found
        rLine.push(prevCnt)
        rLine.sort((c0, c1) => c0.rect.x - c1.rect.x)
        rLine = rLine.map(c => newPiece(toSquare(c.rect)))
        rectsArr.push(rLine)
        rLine = []
      } else {
        rLine.push(prevCnt)
      }
      if (cntArr.indexOf(currCnt) >= cntArr.length - 1) { // last piece
        rLine.push(currCnt)
        rLine.sort((c0, c1) => c0.rect.x - c1.rect.x)
        rLine = rLine.map(c => newPiece(toSquare(c.rect)))
        rectsArr.push(rLine)
      }
      prevCnt = currCnt
    }
  })

  // -> Returning result
  return new params.PipelineResult(rectsArr, src, isRotated)
}

module.exports = {
  runMain
}
