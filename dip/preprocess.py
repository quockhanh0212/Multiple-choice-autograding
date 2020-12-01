from utils import *
import cv2



def toBinary(img, kernel_size):
    blurred = blur_image(img, kernel_size)
    binary = binary_convert(blurred)
    binary = 255 - binary
    return binary

def threshold(binary):
     return cv2.adaptiveThreshold(binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)

def findContour(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    approx = cv2.approxPolyDP(contours[1], 0.01 * cv2.arcLength(contours[1], True), True)
    rect = cv2.minAreaRect(contours[1])
    box = cv2.boxPoints(rect);
    return approx,box

def turnning( approx,box, image):
    corner = find_corner_by_rotated_rect(box, approx)
    image = four_point_transform(image, corner)
    return image


