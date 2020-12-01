from utils import *

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

def find_max_contour(img):
    # Lấy ra vị trí các contour (contour là 1 chuỗi tọa độ (x_i, y_) các điểm có cùng độ sáng dọc theo biên của 1 object (t đang hiểu là thế))
    contours, _ = cv2.findContours(img.astype(np.uint8),  # pylint: disable=unused-variable
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    num = 0
    maxContour = 0
    maxContourArray = []
    for index, cnt in enumerate(contours):
        num += 1
        sizeOfContour = np.array(cnt).shape[0]
        if sizeOfContour > maxContour:
            maxContour = sizeOfContour
            maxContourArray = cnt
    # Vẽ lại contour lớn nhất
    newArray = np.zeros(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(0, h):
        for j in range(0, w):
            for k in maxContourArray:
                if k[0][0] == i and k[0][1] == j:
                    newArray[j][i] = 255
    return newArray


def check_cross(img):
    # Lấy contour lớn nhất
    maxContourArray = find_max_contour(img)
    # Fill contour lớn nhất
    clean_image = binary_fill_holes(maxContourArray)
    clean_image = np.float32(clean_image)
    # Opening để xoá mất gạch
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(clean_image, cv2.MORPH_OPEN, kernel)
    # Trừ bản trước và sau opening để xác định gạch
    minus = clean_image - opening
    count = 0
    # Đếm số lượng pixel để xem có bị gạch không, khá lớn là có bị gạch
    h = img.shape[0]
    w = img.shape[1]
    for i in range(0, h):
        for j in range(0, w):
            if minus[j][i] > 0:
                count += 1
    # Chỉnh ngưỡng nếu cần
    if count > 10:
        return True
    return False

def get_circles_no_cross(img, centers):
    frames = get_frames(img, centers)
    circleNoCross = []
    for i in frames:
        if check_cross(np.array(i)) == False:
            circleNoCross.append(i)
    return circleNoCross