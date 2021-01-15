import os
import random
import numpy as np
import cv2 as cv
import params


# CLASSES
class YoloResultSample:
    def __init__(self, p_id, p_confidence):
        self.id = p_id
        self.confidence = p_confidence


class TemplateResultSample:
    def __init__(self, name="", val_loc=(None, None, None, None)):
        self.piece_name = name
        self.min_val = val_loc[0]
        self.max_val = val_loc[1]
        self.min_loc = val_loc[2]
        self.max_loc = val_loc[3]


class FeatureMatchingResultSample:
    def __init__(self, name="", best_matches=[]):
        self.piece_name = name
        self.matches = best_matches


# FUNCTIONS
def draw_result(src_img, pieces_lst=[], rects_lst=[]):
    for row in range(len(pieces_lst)):
        for tile in range(len(pieces_lst[row])):
            tile_name = pieces_lst[row][tile]
            x = rects_lst[row][tile].x
            y = rects_lst[row][tile].y
            w = rects_lst[row][tile].width
            h = rects_lst[row][tile].height
            color = (255*random.random(), 255*random.random(), 255*random.random())
            cv.rectangle(src_img, (x, y), (x+w, y+h), color, 6)
            cv.putText(src_img, tile_name, (x, y), 2, 1.8, color, 6, cv.LINE_4)
    cv.imshow("final_result", cv.resize(src_img.copy(), dsize=(-1, -1), fx=0.3, fy=0.3))
    # cv.imwrite("../../use_case_images/img-6.jpg", src_img)
    cv.waitKey()
    cv.destroyAllWindows()


def print_result(p_lst=params.PiecesList()):
    print("\n Pieces found:", p_lst.num_pieces)
    str_print = "\n "
    for i in range(len(p_lst.list)):
        for p in p_lst.list[i]:
            str_print += "| %s " % p
        str_print += "|\n "
    print(str_print)


def run_hog_svm(svm, hog, config, res):
    rects_lst = res[0]
    src_img = res[1]
    is_rotated = res[2]
    p_line = []
    pieces_lst = []
    num_pieces = 0
    for r_line in rects_lst:
        for r in r_line:
            roi = src_img[r.y: r.y + r.height, r.x: r.x + r.width]
            resized_roi = cv.resize(roi,
                                    (config.size.width, config.size.height),
                                    cv.INTER_AREA)
            sample = []
            histograms = hog.compute(resized_roi)
            sample.append(histograms)
            sample = np.float32(sample)
            label = svm.predict(sample)
            label = int(label[1][0])
            type_idx = int(np.floor(label / 100))
            piece_idx = int(label % 100)
            piece = config.classes[type_idx].split("\n")[0].split(":")[1].split(",")[piece_idx]
            p_line.append(piece)
            num_pieces += 1
        pieces_lst.append(p_line)
        p_line = []
    return params.PiecesList(pieces_lst, num_pieces, is_rotated)


def run_yolo(net, config, res):
    rects_lst = res[0]
    src_img = res[1]
    is_rotated = res[2]
    p_line = []
    pieces_lst = []
    num_pieces = 0
    for r_line in rects_lst:
        for r in r_line:
            offset = round(r.height / 10)
            roi = src_img[r.y - offset: r.y + r.height + 2 * offset, r.x - offset: r.x + r.width + 2 * offset]
            resized_roi = cv.resize(roi,
                                    (config.size.width, config.size.height),
                                    cv.INTER_CUBIC)

            # Create blob
            blob = cv.dnn.blobFromImage(resized_roi, 1 / 255, (config.size.width, config.size.height),
                                        [0, 0, 0], 1, crop=False)

            # Sets the input to the network
            net.setInput(blob)

            # Runs forward pass to get the output of the output layers
            layers_names = net.getLayerNames()
            outs = net.forward([layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])

            num_pieces += 1
            detected_classes = []

            # Scan through all the bounding boxes output from the network and keep only the ones with high
            # confidence scores
            # Assign the box's class label as the class with the highest score.
            for out in outs:
                for detection in out:
                    if detection[4] > config.object_threshold:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        class_confidence = scores[class_id]
                        if class_confidence > 0:
                            detected_classes.append(YoloResultSample(class_id, class_confidence))
            if len(detected_classes) > 0:
                detected_classes.sort(key=lambda yr: yr.confidence, reverse=True)
                piece_class = config.classes[detected_classes[0].id]
                p_line.append(piece_class)
            else:
                p_line.append("--")
        pieces_lst.append(p_line)
        p_line = []
    return params.PiecesList(pieces_lst, num_pieces, is_rotated)


def run_ssd(sess, config, res):
    rects_lst = res[0]
    src_img = res[1]
    is_rotated = res[2]
    p_line = []
    pieces_lst = []
    num_pieces = 0
    for r_line in rects_lst:
        for r in r_line:
            # Preprocess ROI
            roi = src_img[r.y: r.y + r.height, r.x: r.x + r.width]
            resized_roi = cv.resize(roi,
                                    (config.size.width, config.size.height),
                                    cv.INTER_CUBIC)
            resized_roi = resized_roi[:, :, [2, 1, 0]]  # BGR2RGB

            # Run the model
            out = sess.run([
                sess.graph.get_tensor_by_name("num_detections:0"),
                sess.graph.get_tensor_by_name("detection_scores:0"),
                sess.graph.get_tensor_by_name("detection_boxes:0"),
                sess.graph.get_tensor_by_name("detection_classes:0")],
                feed_dict={
                    "image_tensor:0": resized_roi.reshape(1, resized_roi.shape[0], resized_roi.shape[1], 3)}
            )
            num_pieces += 1
            if out[0][0] > 0:   # if there are detections
                p_line.append(config.classes[int(out[3][0][0]) - 1])  # offset of '1' needed
            else:
                p_line.append("--")
        pieces_lst.append(p_line)
        p_line = []
    return params.PiecesList(pieces_lst, num_pieces, is_rotated)


def run_vgg16(model, config, res, tf):
    rects_lst = res[0]
    src_img = res[1]
    is_rotated = res[2]
    p_line = []
    pieces_lst = []
    num_pieces = 0
    for r_line in rects_lst:
        for r in r_line:
            roi = src_img[r.y: r.y + r.height, r.x: r.x + r.width]
            resized_roi = cv.resize(roi,
                                    (config.size.width, config.size.height),
                                    cv.INTER_CUBIC)
            resized_roi = cv.cvtColor(resized_roi, cv.COLOR_BGR2RGB)
            tensor = tf.convert_to_tensor(resized_roi, dtype=tf.float32)
            tensor = tf.expand_dims(tensor, 0)
            prediction = model.predict(tensor)
            class_id = np.argmax(prediction)
            num_pieces += 1
            if prediction[0][class_id] > 0:
                p_line.append(config.classes[class_id])
            else:
                p_line.append("--")
        pieces_lst.append(p_line)
        p_line = []
    return params.PiecesList(pieces_lst, num_pieces, is_rotated)


def run_vgg19(model, config, res, tf):
    rects_lst = res[0]
    src_img = res[1]
    is_rotated = res[2]
    p_line = []
    pieces_lst = []
    num_pieces = 0
    for r_line in rects_lst:
        for r in r_line:
            roi = src_img[r.y: r.y + r.height, r.x: r.x + r.width]
            resized_roi = cv.resize(roi,
                                    (config.size.width, config.size.height),
                                    cv.INTER_CUBIC)
            resized_roi = cv.cvtColor(resized_roi, cv.COLOR_BGR2RGB)
            tensor = tf.convert_to_tensor(resized_roi, dtype=tf.float32)
            tensor = tf.expand_dims(tensor, 0)
            prediction = model.predict(tensor)
            class_id = np.argmax(prediction)
            num_pieces += 1
            if prediction[0][class_id] > 0:
                p_line.append(config.classes[class_id])
            else:
                p_line.append("--")
        pieces_lst.append(p_line)
        p_line = []
    return params.PiecesList(pieces_lst, num_pieces, is_rotated)


def run_mobilenet_v2(model, config, res, tf):
    rects_lst = res[0]
    src_img = res[1]
    is_rotated = res[2]
    p_line = []
    pieces_lst = []
    num_pieces = 0
    for r_line in rects_lst:
        for r in r_line:
            roi = src_img[r.y: r.y + r.height, r.x: r.x + r.width]
            resized_roi = cv.resize(roi,
                                    (config.size.width, config.size.height),
                                    cv.INTER_CUBIC)
            resized_roi = cv.cvtColor(resized_roi, cv.COLOR_BGR2RGB)
            tensor = tf.convert_to_tensor(resized_roi, dtype=tf.float32)
            tensor = tf.expand_dims(tensor, 0)
            prediction = model.predict(tensor)
            class_id = np.argmax(prediction)
            num_pieces += 1
            if prediction[0][class_id] > 0:
                p_line.append(config.classes[class_id])
            else:
                p_line.append("--")
        pieces_lst.append(p_line)
        p_line = []
    return params.PiecesList(pieces_lst, num_pieces, is_rotated)


def run_resnet152(model, config, res, tf):
    rects_lst = res[0]
    src_img = res[1]
    is_rotated = res[2]
    p_line = []
    pieces_lst = []
    num_pieces = 0
    for r_line in rects_lst:
        for r in r_line:
            roi = src_img[r.y: r.y + r.height, r.x: r.x + r.width]
            resized_roi = cv.resize(roi,
                                    (config.size.width, config.size.height),
                                    cv.INTER_CUBIC)
            resized_roi = cv.cvtColor(resized_roi, cv.COLOR_BGR2RGB)
            tensor = tf.convert_to_tensor(resized_roi, dtype=tf.float32)
            tensor = tf.expand_dims(tensor, 0)
            prediction = model.predict(tensor)
            class_id = np.argmax(prediction)
            num_pieces += 1
            if prediction[0][class_id] > 0:
                p_line.append(config.classes[class_id])
            else:
                p_line.append("--")
        pieces_lst.append(p_line)
        p_line = []
    return params.PiecesList(pieces_lst, num_pieces, is_rotated)


def run_template_matching(tm_type, config, res):
    rects_lst = res[0]
    src_img = res[1]
    is_rotated = res[2]
    p_line = []
    pieces_lst = []
    num_pieces = 0
    template_lst = sorted(os.listdir(config.folder))
    matches_lst = []
    for r_line in rects_lst:
        for r in r_line:
            roi = src_img[r.y: r.y + r.height, r.x: r.x + r.width]
            resized_roi = cv.resize(roi,
                                    (config.size.width, config.size.height),
                                    cv.INTER_AREA)
            for tmpl in template_lst:
                tmpl_path = os.path.join(config.folder, tmpl)
                template = cv.imread(tmpl_path)
                if tm_type == "TM_CCOEFF":
                    match_res = cv.matchTemplate(resized_roi, template, cv.TM_CCOEFF)
                elif tm_type == "TM_SQDIFF":
                    match_res = cv.matchTemplate(resized_roi, template, cv.TM_SQDIFF)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match_res)
                matches_lst.append(TemplateResultSample(tmpl.split("-")[0], (min_val, max_val, min_loc, max_loc)))
            if tm_type == "TM_CCOEFF":
                matches_lst.sort(key=lambda t: t.max_val, reverse=True)
            elif tm_type == "TM_SQDIFF":
                matches_lst.sort(key=lambda t: t.min_val, reverse=False)
            p_line.append(matches_lst[0].piece_name)
            num_pieces += 1
            matches_lst = []
        pieces_lst.append(p_line)
        p_line = []
    return params.PiecesList(pieces_lst, num_pieces, is_rotated)


def run_features_det_desc_match(fdd, bf, config, res):
    rects_lst = res[0]
    src_img = res[1]
    is_rotated = res[2]
    p_line = []
    pieces_lst = []
    num_pieces = 0
    template_lst = sorted(os.listdir(config.folder))
    matches_lst = []
    for r_line in rects_lst:
        for r in r_line:
            roi = src_img[r.y: r.y + r.height, r.x: r.x + r.width]
            resized_roi = cv.resize(roi,
                                    (config.size.width, config.size.height),
                                    cv.INTER_CUBIC)
            piece_kps, piece_desc = fdd.detectAndCompute(resized_roi, None)
            for tmpl in template_lst:
                tmpl_path = os.path.join(config.folder, tmpl)
                template = cv.imread(tmpl_path)
                template_kps, template_desc = fdd.detectAndCompute(template, None)
                knn_matches = bf.knnMatch(piece_desc, template_desc, 1)
                good = []
                for match in knn_matches:
                    if match:
                        good.append(match)
                piece_points = np.zeros((len(good), 2), dtype=np.float32)
                template_points = np.zeros((len(good), 2), dtype=np.float32)
                for i, g in enumerate(good):
                    piece_points[i, :] = piece_kps[g[0].queryIdx].pt
                    template_points[i, :] = template_kps[g[0].trainIdx].pt
                h, mask = cv.findHomography(piece_points, template_points, cv.RANSAC, 5.0)
                best = []
                for i in range(len(mask)):
                    if mask[i][0] == 1:
                        best.append(good[i])
                matches_lst.append(FeatureMatchingResultSample(tmpl.split("-")[0], best))
            matches_lst.sort(key=lambda x: len(x.matches), reverse=True)
            p_line.append(matches_lst[0].piece_name)
            num_pieces += 1
            matches_lst = []
        pieces_lst.append(p_line)
        p_line = []
    return params.PiecesList(pieces_lst, num_pieces, is_rotated)
