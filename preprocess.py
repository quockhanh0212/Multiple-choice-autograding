from utils import *

def preprocess(img):
    binary = binary_convert(img)
    binary = 255 - binary
    binary = cv2.resize(binary, (650, 800))
    return binary
