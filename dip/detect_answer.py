import cv2
import numpy as np

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

def sort_choices_order(choices, method="top-to-bottom"):
	pass

def template_matching():
    pass

def get_answer(choices):
    answers = []
    # TYPE YOUR CODE HERE
    return answers



