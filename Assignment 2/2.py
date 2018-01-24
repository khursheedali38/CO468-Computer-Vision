# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:01:25 2018

@author: Khursheed Ali
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
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
    #src image
    img = cv2.imread('stich2.jpg', 0)
    Z = harris(img)
    Z = cv2.dilate(Z,None)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    color_img[Z>0.9999999999*Z.max()]=[0,0,255]
    
    #save cordinates of corner points of src image
    corners_list = []
    corneri, cornerj = np.where(Z>0.9999999999*Z.max())
    for i in range(corneri.shape[0]):
            corners_list.append([corneri[i], cornerj[i]])
            
    plt.subplot(1, 2, 1),plt.imshow(img, cmap='gray')

      
    #dst image    
    img = cv2.imread('stich1.jpg', 0)
    Z = harris(img)
    Z = cv2.dilate(Z,None)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    color_img[Z>0.9999999999*Z.max()]=[0,0,255]
            
    #save cordinates of corner points of dst image
    corners_list1 = []
    corneri, cornerj = np.where(Z>0.9999999999*Z.max())
    for i in range(corneri.shape[0]):
            corners_list1.append([corneri[i], cornerj[i]])
            
    plt.subplot(1, 2, 2),plt.imshow(img, cmap='gray')
    
    plt.show()
    M, mask = cv2.findHomography(corners_list, corners_list1, cv2.RANSAC, 5.0)

    #create a canvas
    m = 2000
    n = 2000
    M = np.float32([[1,0,100],[0,1,200]])
    dst = cv2.warpAffine(img,M,(m,n))
    cv2.imwrite('affine.jpg',dst)
    


    
