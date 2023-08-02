import cv2 
import numpy as np

# Load image
img = cv2.imread('disclaimer-lg.jpg')

# Background color
bg_color = (32, 29, 28)  

# Preprocess
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] 

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

invert = 255 - opening

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10)) 
dilated = cv2.dilate(invert, kernel, iterations=5)

# Find contours
contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = contours

for c in cnts:

  mask = np.zeros(img.shape[:2], np.uint8)
  cv2.drawContours(mask, [c], -1, 255, -1)  

  # bright = (mask[:,:,:] > bg_color).all(axis=2)
  bright = (mask[:,:] > bg_color[0]).all()


  x,y,w,h = cv2.boundingRect(c)
  ar = w / float(h)

  variance = cv2.meanStdDev(mask[:,:])[1] 

  if bright and ar > 3.5 and variance < 15:
    cv2.fillPoly(img, [c], bg_color)

# Show result  
cv2.imshow('Image', img)
cv2.waitKey(0)