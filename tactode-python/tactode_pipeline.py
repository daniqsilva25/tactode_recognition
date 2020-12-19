import cv2 as cv
import numpy as np
import math


# CLASSES
class Center:
    def __init__(self, xx=-1, yy=-1):
        self.x = xx
        self.y = yy


class Rect:
    def __init__(self, xx=-1, yy=-1, hc=-1, wc=-1):
        self.x = int(xx)
        self.y = int(yy)
        self.height = int(hc)
        self.width = int(wc)


class Contour:
    def __init__(self, ic=-1, hc=None, r=Rect(), cc=Center(),
                 ar=None, per=None, asp_rat=None, ext=None, equi_diam=None):
        self.idx = ic
        self.have_teeth = False
        self.hierarchy = hc
        self.rect = r
        self.center = cc
        self.area = ar
        self.perimeter = per
        self.aspect_ratio = asp_rat
        self.extent = ext
        self.equi_diameter = equi_diam

    def update_center(self):
        self.center.x = self.rect.x + round(self.rect.width / 2)
        self.center.y = self.rect.y + round(self.rect.height / 2)


# FUNCTIONS
def draw_contours(img, arr, contours):
    for e in range(len(arr)):
        img = cv.drawContours(img, contours, arr[e].idx, (0, 255, 0), 6)
    return img


def draw_rects(img, arr):
    for r_elem in arr:
        cv.rectangle(img, (r_elem.rect.x, r_elem.rect.y),
                     (r_elem.rect.x + r_elem.rect.width, r_elem.rect.y + r_elem.rect.height),
                     (0, 255, 255), 6)
    return img


def to_square(rect=Rect()):
    diff = round(abs(rect.width - rect.height) / 2)
    side = rect.width if rect.width >= rect.height else rect.height
    x_r = rect.x if rect.width >= rect.height else rect.x - diff
    y_r = rect.y if rect.height >= rect.width else rect.y - diff
    return Rect(x_r, y_r, side, side)


def union_rect(r1=Rect(), r2=Rect()):
    top1 = r1.y
    bottom1 = r1.y + r1.height
    left1 = r1.x
    right1 = r1.x + r1.width
    top2 = r2.y
    bottom2 = r2.y + r2.height
    left2 = r2.x
    right2 = r2.x + r2.width

    top = top1 if top1 <= top2 else top2
    bottom = bottom1 if bottom1 >= bottom2 else bottom2
    left = left1 if left1 <= left2 else left2
    right = right1 if right1 >= right2 else right2

    width = right - left
    height = bottom - top

    return Rect(left, top, height, width)


def run_main(img_path, hog, svm, config):
    # Reading input image
    src = cv.imread(img_path)
    old_height = src.shape[0]
    old_width = src.shape[1]

    # Resizing image
    norm_size = 1920
    new_height, new_width, aspect_ratio = 0, 0, 0
    if old_height != norm_size and old_width != norm_size:
        if old_height >= old_width:
            aspect_ratio = old_height / old_width
            new_height = norm_size
            new_width = int(np.floor(norm_size / aspect_ratio))
        else:
            aspect_ratio = old_width / old_height
            new_width = norm_size
            new_height = int(np.floor(norm_size / aspect_ratio))
        src = cv.resize(src, (new_width, new_height), cv.INTER_AREA)
    # cv.imshow("resized", cv.resize(src.copy(), dsize=(0, 0), fx=0.3, fy=0.3))
    # cv.waitKey()
    # cv.imwrite("../../use_case_images/img-1.jpg", src)

    # Adjusting image
    to_adjust = src.copy()
    to_adjust = cv.GaussianBlur(to_adjust, (9, 9), 0)
    to_adjust = cv.resize(to_adjust, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA)
    to_adjust = cv.cvtColor(to_adjust, cv.COLOR_BGR2GRAY)
    to_adjust = cv.Canny(to_adjust, 100, 200, apertureSize=3)
    hough_lines = cv.HoughLines(to_adjust, rho=1, theta=np.pi / 180, threshold=50,
                                srn=0, stn=0, min_theta=0, max_theta=np.pi)

    if hough_lines.shape[0] > 0:
        angle = -1
        big_angles = []
        found_biggest_lines = False
        for row in range(0, len(hough_lines)):
            for rho, theta in hough_lines[row]:
                degrees = theta * 180 / np.pi
                if angle == -1:
                    angle = degrees
                    big_angles.append(degrees)
                else:
                    if 40 < np.abs(degrees - angle) < 140:
                        big_angles.append(degrees)
                        found_biggest_lines = True
            if found_biggest_lines:
                break

    for a in big_angles:
        diff_angle = a
        idx_big_angle = big_angles.index(a)
        if 45 < diff_angle <= 135:
            diff_angle -= 90
        elif diff_angle > 135:
            diff_angle -= 180
        else:
            diff_angle += 0
        big_angles.pop(idx_big_angle)
        big_angles.insert(idx_big_angle, diff_angle)

    if len(big_angles) > 1:
        if big_angles[0] * big_angles[1] < 0:
            adjusting_angle = (big_angles[0] - big_angles[1]) / 2
        else:
            adjusting_angle = (big_angles[0] + big_angles[1]) / 2
    else:
        adjusting_angle = big_angles[0]

    h = src.shape[0]
    w = src.shape[1]
    rangle = np.deg2rad(adjusting_angle)    # angle in radians
    # Compute new image height and width
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * 1.
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * 1.
    # Compute rotation matrix
    rot_mat = cv.getRotationMatrix2D((nw*.5, nh*.5), adjusting_angle, 1.)
    # Compute the move from the old center to the new center combined w/ no rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*.5, (nh-h)*.5, 0]))
    # Update the translation
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    src = cv.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv.INTER_LANCZOS4)
    # cv.imshow("adjusted", cv.resize(src.copy(), dsize=(0, 0), fx=0.3, fy=0.3))
    # cv.waitKey()
    # cv.imwrite("../../use_case_images/img-2.jpg", src)

    # Binarising image
    blur = cv.GaussianBlur(src, (11, 11), 0)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (6, 6))
    grad = cv.morphologyEx(blur, cv.MORPH_GRADIENT, kernel)
    bw = cv.inRange(grad, (0, 0, 0), (20, 20, 20))
    fg = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, kernel)
    # cv.imshow("binary", cv.resize(fg, dsize=(0, 0), fx=0.3, fy=0.3))
    # cv.waitKey()
    # cv.imwrite("../../use_case_images/img-3.jpg", fg)

    # Segmenting and rotating image
    rotation_step = 0
    is_rotated = True
    cnt_arr = []
    child_cnt_arr = []
    teeth_arr = []
    biggest_cnt = Contour()

    while rotation_step < 4 and is_rotated:
        biggest_cnt = Contour()
        smallest_cnt = Contour()
        cnt_arr.clear()

        # Finding contours
        contours, hierarchy = cv.findContours(fg, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        # immm, contours, hierarchy = cv.findContours(fg, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # Getting contours' properties
        for i in range(0, len(contours)):
            cnt = contours[i]
            hier = hierarchy[0][i]

            x, y, w, h = cv.boundingRect(cnt)

            rect_area = int(w * h)
            center = Center(int(x + w / 2), int(y + h / 2))

            area = int(cv.contourArea(cnt))
            perimeter = cv.arcLength(cnt, True)
            cnt_aspect_ratio = w / h
            extent = area / rect_area
            equi_diameter = np.sqrt(4 * area / np.pi)

            cnt_arr.append(Contour(i, hier, Rect(x, y, h, w), center,
                                   area, perimeter, cnt_aspect_ratio, extent, equi_diameter))

        # Finding grandparent, parent and child contours
        cnt_arr.sort(key=lambda c: c.area, reverse=True)
        grandparent_idx = -1
        parent_idx = -1
        for c in cnt_arr:
            ch = c.hierarchy
            ci = c.idx
            if grandparent_idx == -1:
                if ch[3] == -1 and ch[2] != -1:
                    grandparent_idx = ci
            else:
                if ch[3] == grandparent_idx:
                    parent_idx = ci
                    break

        child_cnt_arr.clear()
        for c in cnt_arr:
            if c.hierarchy[3] == parent_idx:
                child_cnt_arr.append(c)

        # Finding biggest and smallest tile-alike contours
        for i in range(0, len(child_cnt_arr)):
            c = child_cnt_arr[i]
            if biggest_cnt.idx == -1 and c.aspect_ratio >= 0.7:
                biggest_cnt = c
            elif biggest_cnt != -1 and c.area / biggest_cnt.area < 0.35:
                smallest_cnt = child_cnt_arr[i - 1]
                break

        # Detecting teeth in the tiles
        for i in range(child_cnt_arr.index(biggest_cnt), child_cnt_arr.index(smallest_cnt) + 1):
            cnt = child_cnt_arr[i]
            width = int(cnt.rect.width)
            height = round(cnt.rect.height / 3)
            x = int(cnt.rect.x)
            y = int(cnt.rect.y + 2 * height)
            height = int(round(width / 2))
            roi = src[y: y + height, x: x + width]
            roi = cv.resize(roi, (config.size.width, config.size.height), cv.INTER_AREA)
            sample = []
            histograms = hog.compute(roi)
            sample.append(histograms)
            sample = np.float32(sample)
            label = svm.predict(sample)
            label = int(label[1][0])
            cnt.have_teeth = True if label == 1 else False

        # Filtering teeth detections and identifying rotation
        teeth_arr.clear()
        for c in child_cnt_arr:
            if c.have_teeth:
                teeth_arr.append(c)
        if len(teeth_arr) > 0:
            teeth_arr.sort(key=lambda c: c.rect.x, reverse=False)
            most_left_tile = teeth_arr[0]
            tiles_at_left = False
            for c in child_cnt_arr:
                if c.rect.x < most_left_tile.rect.x - most_left_tile.rect.width / 2:
                    tiles_at_left = True
                    break
            if not tiles_at_left:
                prev_teeth = Contour()
                column_count = 1
                for curr_teeth in teeth_arr:
                    if prev_teeth.idx == -1:
                        prev_teeth = curr_teeth
                    else:
                        if np.abs(prev_teeth.rect.x - curr_teeth.rect.x) <= prev_teeth.rect.width / 2:
                            column_count += 1
                            prev_teeth = curr_teeth
                        if column_count >= 2:
                            is_rotated = False
                            break
        if is_rotated:
            fg = cv.rotate(fg, cv.ROTATE_90_CLOCKWISE)
            src = cv.rotate(src, cv.ROTATE_90_CLOCKWISE)
        rotation_step += 1
    # cv.imshow("rotated", cv.resize(draw_contours(src.copy(), child_cnt_arr, contours),
    #                               dsize=(0, 0), fx=0.3, fy=0.3))
    # cv.waitKey()
    # cv.imwrite("../../use_case_images/img-4.jpg", draw_contours(src.copy(), child_cnt_arr, contours))

    # Removing inner and noisy contours
    sub_arr = child_cnt_arr.copy()
    for i in range(0, len(child_cnt_arr)):
        curr_cnt = child_cnt_arr[i]
        if curr_cnt.idx != -1:
            curr_rect_area = int(curr_cnt.rect.width * curr_cnt.rect.height)
            for j in range(i + 1, len(sub_arr)):
                sub_cnt = sub_arr[j]
                if sub_cnt.idx != -1:
                    result_union_rect = union_rect(curr_cnt.rect, sub_cnt.rect)
                    result_union_rect_area = int(result_union_rect.width * result_union_rect.height)
                    if result_union_rect_area == curr_rect_area or \
                            sub_cnt.aspect_ratio < 0.25 or sub_cnt.aspect_ratio > 5 or \
                            sub_cnt.area / biggest_cnt.area < 0.01 or sub_cnt.extent <= 0.25:
                        child_cnt_arr[j].idx = -1

    aux_cnt_arr = []
    for c in child_cnt_arr:
        if c.idx != -1:
            aux_cnt_arr.append(c)
    child_cnt_arr.clear()
    child_cnt_arr = aux_cnt_arr.copy()
    aux_cnt_arr.clear()
    # cv.imshow("without noisy cnts", cv.resize(draw_contours(src.copy(), child_cnt_arr, contours),
    #                                          dsize=(0, 0), fx=0.3, fy=0.3))
    # cv.waitKey()

    # Agglomerating little bits of remaining contours
    def sort_by_center_distance(ccd=Contour()):
        cx = pow((ccd.center.x - curr_center.x), 2)
        cy = pow((ccd.center.y - curr_center.y), 2)
        dist = round(np.sqrt(cx + cy))
        return dist

    sub_arr.clear()
    child_cnt_arr.sort(key=lambda c: c.area, reverse=False)
    sub_arr = child_cnt_arr.copy()
    for i in range(0, len(child_cnt_arr)):
        curr_cnt = child_cnt_arr[i]
        curr_rect_area = curr_cnt.rect.width * curr_cnt.rect.height
        curr_center = curr_cnt.center
        sub_sub_arr = [Contour()]
        idx = -1
        for j in range(i + 1, len(sub_arr)):
            sub_cnt = sub_arr[j]
            sub_rect_area = sub_cnt.rect.width * sub_cnt.rect.height
            if curr_rect_area / sub_rect_area < 0.25:
                sub_sub_arr = sub_arr[j:]
                idx = j
                break
        if idx != -1:
            child_cnt_arr[i].idx = -1
            sub_sub_arr.sort(key=sort_by_center_distance, reverse=False)
            closest_cnt = Contour()
            next_cnt = False
            for k in range(0, len(sub_sub_arr)):
                left_limit = sub_sub_arr[k].rect.x
                right_limit = sub_sub_arr[k].rect.x + sub_sub_arr[k].rect.width
                top_limit = sub_sub_arr[k].rect.y
                bottom_limit = sub_sub_arr[k].rect.y + sub_sub_arr[k].rect.height
                if left_limit <= curr_center.x <= right_limit:
                    if curr_center.y < sub_sub_arr[k].center.y or (top_limit <= curr_center.y <= bottom_limit):
                        closest_cnt = sub_sub_arr[k]
                        break
                    elif next_cnt:
                        closest_cnt = sub_sub_arr[k - 1]
                        break
                    else:
                        next_cnt = True
                else:
                    if top_limit <= curr_center.y <= bottom_limit:
                        closest_cnt = sub_sub_arr[k]
                        break
                    elif next_cnt:
                        closest_cnt = sub_sub_arr[k - 1]
                        break
                    else:
                        next_cnt = True
            other_idx = child_cnt_arr.index(closest_cnt)
            if other_idx != -1:
                child_cnt_arr[other_idx].rect = union_rect(child_cnt_arr[other_idx].rect, curr_cnt.rect)
                child_cnt_arr[other_idx].update_center()

    for c in child_cnt_arr:
        if c.idx != -1:
            aux_cnt_arr.append(c)
    child_cnt_arr.clear()
    child_cnt_arr = aux_cnt_arr.copy()
    aux_cnt_arr.clear()

    # cv.imwrite("out.jpg", draw_rects(src.copy(), child_cnt_arr))
    # print("Number of tiles found: ", len(child_cnt_arr))
    # cv.imshow("agglomerated", cv.resize(draw_contours(src.copy(), child_cnt_arr, contours),
    #                             dsize=(0, 0), fx=0.3, fy=0.3))
    # cv.waitKey()
    # cv.imwrite("../../use_case_images/img-5.jpg", draw_contours(src.copy(), child_cnt_arr, contours))

    # Positioning
    child_cnt_arr.sort(key=lambda c: c.rect.y, reverse=False)
    tiles = []
    line = []
    prev_cnt = Contour()
    for curr_cnt in child_cnt_arr:
        if prev_cnt.idx == -1:
            prev_cnt = curr_cnt
        else:
            if curr_cnt.rect.y > prev_cnt.rect.y + prev_cnt.rect.height / 4:
                line.append(to_square(prev_cnt.rect))
                line.sort(key=lambda r: r.x, reverse=False)
                aux_line = line.copy()
                tiles.append(aux_line)
                line.clear()
            else:
                line.append(to_square(prev_cnt.rect))
            if child_cnt_arr.index(curr_cnt) >= len(child_cnt_arr) - 1:
                line.append(to_square(curr_cnt.rect))
                line.sort(key=lambda r: r.x, reverse=False)
                aux_line = line.copy()
                tiles.append(aux_line)
            prev_cnt = curr_cnt
    return tiles, src, is_rotated
