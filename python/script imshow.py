# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
cv2.__version__
import numpy as np

img = cv2.imread('img/Guido_Reni_031.jpg', cv2.IMREAD_COLOR)

cv2.imshow('img',img)

while(True):
    if cv2.waitKey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()