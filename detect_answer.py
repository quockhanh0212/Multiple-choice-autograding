import cv2
import numpy as np
from PIL import Image

def get_frames(img, centers):
    frames = []
    l = 15
    immm = img.astype(np.uint8)

    for i in centers:
        a = []
        for y in range(i[1] - l, i[1] + l):
            row = []
            for x in range(i[0] - l, i[0] + l):
                row.append(immm[y][x])
            a.append(row)
        frames.append(a)

    return frames

def convert_to_binary_img(img):
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                      cv2.THRESH_BINARY, 31, 7)
    return img

def convert_list_to_array(obj):
    array = np.array(obj, dtype=np.uint8)
    new_image= Image.fromarray(array)
    return np.asarray(new_image)

def template_matching():
    arrayTemplate = []
    templateA = cv2.imread('exam2_A.png', cv2.IMREAD_GRAYSCALE)

    templateA = cv2.adaptiveThreshold(templateA, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                      cv2.THRESH_BINARY, 31, 7)
    templateB = cv2.imread('exam2_B.png', cv2.IMREAD_GRAYSCALE)
    templateB = cv2.adaptiveThreshold(templateB, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                      cv2.THRESH_BINARY, 31, 7)
    templateC = cv2.imread('exam2_C.png', cv2.IMREAD_GRAYSCALE)
    templateC = cv2.adaptiveThreshold(templateC, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                      cv2.THRESH_BINARY, 31, 7)
    templateD = cv2.imread('exam2_D.png', cv2.IMREAD_GRAYSCALE)
    templateD = cv2.adaptiveThreshold(templateD, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                      cv2.THRESH_BINARY, 31, 7)

    arrayTemplate.append(templateA)
    arrayTemplate.append(templateB)
    arrayTemplate.append(templateC)
    arrayTemplate.append(templateD)
    return arrayTemplate

def get_answer(choices):
    arrayTemplate = template_matching()
    answers = []
    dem = 0
    for j in range(len(choices)):
        min = 0
        k = -1
        for i in range(len(arrayTemplate)):
            choices[j] = convert_to_binary_img(convert_list_to_array(choices[j]))
            result = cv2.matchTemplate(choices[j], arrayTemplate[i], cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if min > min_val:
                min = min_val
                k = i
        answers.append(k)
        dem += 1
        check = 0
    answers.reverse()
    return answers

def convert_answer_type(answers):
    converted_answer = []
    for answer in answers:
        if answer == 'A':
            converted_answer.append(0)
        elif answer == 'B':
            converted_answer.append(1)
        elif answer == 'C':
            converted_answer.append(2)
        else:
            converted_answer.append(3)

    return converted_answer