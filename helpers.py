import cv2 as cv
import numpy as np

def roi(image, x1, x2, y1, y2):
    roi = image[x1:x2, y1:y2]           # reduce image to given rectangle
    b,g,r = cv.split(roi)               # grayscale by just taking the blue channel, can be changed to cv2's grayscale
    return b

def enhance(image, blur_value=7, sharp_value=5):
    eq = cv.equalizeHist(image)                                                 # basic histogram equalization, should try clahe too
#    blur = cv.medianBlur(eq, blur_value)                                       # median blur didn't work out well, keeping the line cause maybe we can experiment with it later
    blur = cv.GaussianBlur(eq, (blur_value, blur_value), 0)                     # gauss blur
    #kernel = np.array([[0, -1, 0], [-1, sharp_value, -1], [0, -1, 0]])         # sharpening kernel
    #sharp = cv.filter2D(blur, -1, kernel)                                      # doesnt improve canny results
    return blur

def kmeans(image, k=3):                                                         # tried k-means over edge detection, not much better and also wasnt covered in class
    pixel_values = image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    res = cv.medianBlur(segmented_image, 3)
    return res

def get_edges(image):                                                               # canny edge detection
    canny = cv.Canny(image, 50, 100)
    contours, _ = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return contours

def threshold(image, threshold=120):                                                # threshold and return region mask
    _, region = cv.threshold(image, threshold, 255, cv.THRESH_OTSU)
    return region

def timeofday(image, threshold=90):                                                 # guess time of day based on mean intensity
    return (cv.mean(image)[0] > threshold)

def show(image, title=""):                                                          # shorthand to show images
    cv.imshow(title, image)
    cv.waitKey()
