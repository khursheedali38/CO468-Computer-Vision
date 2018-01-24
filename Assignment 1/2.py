# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:01:25 2018

@author: Khursheed Ali
"""

import numpy as np
import cv2

def hmn(img):
    kernel = np.array([-1, 0, 1])
    return cv2.filter2D(img, -1, kernel)

def vmn(img):
    kernel = np.array([-1, 0, 1]).reshape(1, 3)
    return cv2.filter2D(img, -1, kernel)

def harris(img, sigma=0.0, k=0.04):
    partialDerX = hmn(img)
    partialDerY = vmn(img)

    A = partialDerX**2
    B = partialDerY**2
    C = partialDerX * partialDerY

    A = cv2.GaussianBlur(A,(7,7),1.4)
    B = cv2.GaussianBlur(B,(7,7),1.4)
    C = cv2.GaussianBlur(C,(7,7),1.4)

    Tr = A + B
    Det = (A*B) - C**2
    R = Det - k*(Tr**2)

    return R

if __name__ == '__main__':
    img = cv2.imread('appmech.jpg', 0)
    Z = harris(img)
    Z = cv2.dilate(Z,None)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    color_img[Z>0.9999999999*Z.max()]=[0,0,255]
    cv2.imwrite('final.jpg',color_img)
    
