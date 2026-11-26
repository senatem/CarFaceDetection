import cv2 as cv
import glob
import numpy as np
import helpers
from helpers import show

def locate_car(image):
    #TODO enhanced pretty good, need to improve edges and extract the car (or window directly? maybe thats easier)
    roi = helpers.roi(image, 200, 900, 600, 1250)       # reduce image to just where the car is likely to be
    enhanced = helpers.enhance(roi)                     # enhance image to extract features easily
    contours = helpers.get_edges(enhanced)              # canny edge detector (contours are too disconnected, need to improve)

    for c in contours:                                  # process found contours
        if cv.contourArea(c) > 100:                     # filter by contour size
            cv.drawContours(enhanced, [c], -1, (0,255,0), 2)

    return enhanced

def locate_window(image):
    # TODO or combine with the above if possible
    return

def locate_face(image):
    # TODO should be easier after locating the window
    # - create mask for window region (can be reduced to just the right half, in fact)
    # - enhance masked area to emphasize the face
    # - draw minrect
    return

path = glob.glob("images/*.jpg")
for img in path:
    image = cv.imread(img)
    daytime = helpers.timeofday(image) # 1->day 0->night
                                       # not used right now, but could be useful to improve night images

    car = locate_car(image)
    show(car)
