def get_frames(img, centers):
    frames = []
    d = 10

    for center in centers:
        y1 = center[0] - d
        y2 = center[0] + d
        x1 = center[1] - d
        x2 = center[1] + d
        frame = img[x1:x2, y1:y2]
        frames.append(frame)

    return frames

def find_max_contour(img):
    pass

def check_cross():
    pass

def get_circles_no_cross(centers):
    pass