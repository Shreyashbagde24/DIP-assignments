from unittest import result
import cv2
import numpy as np

org = cv2.imread('image.jpg')

imageHeight = len(org)
imageWidth = len(org[0])

#Shape1

sub = np.zeros((imageHeight, imageWidth), dtype=np.uint8)

for i in range(250, imageHeight-250):
    for j in range(250, imageWidth-250):
        sub[i][j] = int(255)

#Shape2

image2Height = len(org)
image2Width = len(org[0])

#Shape1

result = np.zeros((image2Height, image2Width), dtype=np.uint8)

for i in range(300, image2Height-300):
    for j in range(100, image2Width-100):
        result[i][j] = int(255)


cv2.imshow('org', org)  
cv2.imshow('Shape_2', result)                
cv2.imshow('Shape_1', sub)
cv2.imshow('And_Operation', result & sub)  
cv2.imshow('OR_Operation', result | sub) 
cv2.imshow('Not_Operation_1', ~sub ) 
cv2.imshow('Not_Operation_2', ~result) 
        
cv2.waitKey(0)