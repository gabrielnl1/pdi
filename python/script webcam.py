# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
cv2.__version__

vid = cv2.VideoCapture(0)

if vid.isOpened():
    print('Webcam OK!')
else:
    print('ERROR: Webcam not found!')
    
while(True):
    _, frame = vid.read()
    cv2.imshow('frame',frame)
    if cv2.waitkey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()
vid.release()