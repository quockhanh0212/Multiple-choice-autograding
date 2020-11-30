from utils import *

def preprocess(img, kernel_size):
    blurred = blur_image(img, kernel_size)
    binary = binary_convert(blurred)
    binary = 255 - binary
    return binary
