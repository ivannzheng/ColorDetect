import cv2
import numpy as np
from util import get_limits


color_input = input("What color would you like me to find? (red, blue, green, yellow, purple, orange): ")

colors = {
    "red": [[0, 255, 255], [170,  255, 255]],
    "blue": [120, 255, 255],
    "green": [60, 255, 255],
    "yellow": [35, 255, 255],
    "purple": [150, 255, 255],
    "orange": [25, 255, 255]
}


color_hsv = colors[color_input]


cap = cv2.VideoCapture(0)


fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

while True:
    ret, frame = cap.read()

    
    fgmask = fgbg.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    
    foreground = cv2.bitwise_and(frame, frame, mask=fgmask)

  
    hsvImage = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)

    
    limits = get_limits(color=color_hsv)

  
    if color_input == "red":
        mask1 = cv2.inRange(hsvImage, limits[0][0], limits[0][1])
        mask2 = cv2.inRange(hsvImage, limits[1][0], limits[1][1])
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lowerLimit, upperLimit = limits
        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # Find contours in the mask, code to only have one box around the largest object with the color, so there wont be tons of boxes off miss-reconginition
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw the bounding box around the largest contour if it's above a certain size
        if w * h > 500: 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




