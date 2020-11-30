from scipy.ndimage.morphology import binary_fill_holes
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    for i in range(right_answers):
        if answers[i] == right_answers[i]:
            score+= sc_each

    return score