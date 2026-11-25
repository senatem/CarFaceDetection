import cv2 as cv
import glob
import numpy as np
import helpers

path = glob.glob("images/*.jpg")
for img in path:
    image = cv.imread(img)
    roi = helpers.roi(image)
    enhanced = helpers.enhance(roi)
    contours = helpers.get_edges(enhanced)

    for c in contours:
        box=cv.minAreaRect(c)
        #print(box)
        box=cv.boxPoints(box)
        box=np.array(box,dtype='int')
        cv.drawContours(enhanced, [c], -1, (0,255,0), 2)

    cv.imshow("Image", enhanced)
    cv.waitKey()
