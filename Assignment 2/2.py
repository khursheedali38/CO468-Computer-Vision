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
            
    plt.subplot(1, 3, 1),plt.imshow(img, cmap='gray')
    
    #writing to file all corner points
    outfile = open('corners_src.txt', 'w')
    for i in range(100):
        outfile.write(str(corners_list[i][0]) + ' ' + str(corners_list[i][1]) + '\n')
    outfile.close()

      
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
            
    plt.subplot(1, 3, 2),plt.imshow(img, cmap='gray')
    
    #wr;iting t file all corne rpints
    outfile = open('corners_dst.txt', 'w')
    for i in range(100):
        outfile.write(str(corners_list1[i][0]) + ' ' + str(corners_list1[i][1]) + '\n')
    outfile.close()
    
    
    #converting to nunpy array
    src_common = [[110.96, 324.813], [94.99, 322.15], [56.4299, 343.433], [43.13, 336.783], [162.83, 282.253], [188.1, 246.343]]
    dst_common = [[499.9, 36.77], [486.611, 358.062], [442.7, 374.02], [433.41, 364.712], [553.11, 316.833], [574.391, 272.943]]
    src_pts = np.asarray(src_common)
    dst_pts = np.asarray(dst_common)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    src_img = cv2.imread('stich2.jpg', 1)
    warped = cv2.warpPerspective(src_img,M,(4000,2000))

    plt.subplot(1, 3, 3), plt.imshow(warped, cmap='gray')

    plt.show()


    


    
