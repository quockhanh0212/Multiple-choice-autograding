from scipy.ndimage.morphology import binary_fill_holes
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def blur_image(img, kernel_size):
    blurred = cv2.GaussianBlur(img, kernel_size, 0)
    plt.imshow(blurred, cmap='gray')
    return blurred

def binary_convert(img):
    binary = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,31,7)
    plt.imshow(binary, cmap='gray')
    return binary


def opening(img, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray')
    return opening

def closing(img, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    plt.imshow(closing, cmap='gray')
    return closing

def fill_holes(img):
    clean_image = binary_fill_holes(img)
    clean_image = np.float32(clean_image)
    plt.imshow(clean_image, cmap='gray')
    return clean_image

def find_contours(img):
    contours, _ = cv2.findContours(img.astype(np.uint8),
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
    return contours


def detect_circle(img, contours, round_thresh, area_thresh):

    plt.imshow(img.astype(np.uint8), cmap='gray')
    centers = []

    for ind, contour in enumerate(contours):
        # tính chu vi của contour
        perimeter = cv2.arcLength(contour, True)
        # tính diện tích của contour
        area = cv2.contourArea(contour)

        # nếu như chu vi bằng 0 hoặc area nhỏ hơn threshold thì ta đặt ngay góc alpha = 0
        if perimeter == 0 or area < area_thresh:
            alpha = 0
        else:
            alpha = 4 * np.pi * area / (perimeter ** 2)

        # Vẽ vòng tròn đỏ tại tâm của contour thỏa mãn điều kiện là vòng tròn
        if alpha > round_thresh:
            moments = cv2.moments(contour)
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            centers.append([cx, cy])

            plt.plot(cx, cy, 'ro')

    plt.show()
    return centers

def get_score(answers, right_answers):
    score = 0
    sc_each = 10/len(right_answers)
    for i in range(len(right_answers)):
        if answers[i] == right_answers[i]:
            score+= sc_each

    return score

def distance(p1, p2):
    my_dist = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    return my_dist


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def find_corner_by_rotated_rect(box, approx):
    corner = []
    for p_box in box:
        min_dist = 999999999
        min_p = None
        for p in approx:
            dist = distance(p_box, p[0])
            if dist < min_dist:
                min_dist = dist
                min_p = p[0]
        corner.append(min_p)

    corner = np.array(corner)
    return corner