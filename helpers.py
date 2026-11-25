import cv2 as cv
import numpy as np

def process(image):
    return image

def roi(image):
    roi = image[200:900, 600:1250]
    b,g,r = cv.split(roi)
    return b

def enhance(image):
    contrast = cv.normalize(image, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
    blur = cv.medianBlur(contrast, 7)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv.filter2D(blur, -1, kernel)
    return sharp

def get_edges(image):
    canny = cv.Canny(image, 50, 150)
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

def locate_window(image):
    return

def locate_face(image):
    return

def timeofday(image, threshold=90):
    if (cv.mean(image)[0] < threshold):
        return "night"
    else:
        return "day"
