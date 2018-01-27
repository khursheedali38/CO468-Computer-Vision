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
    #detect corners
    dst = cv2.imread('stich1.jpg', 1)
    src = cv2.imread('stich2.jpg', 1)
    Z = harris(dst)
    Z = cv2.dilate(Z,None)
    color_img = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
    color_img[Z>0.9999999999*Z.max()]=[0,0,255]

    #canvas image dst
    M = np.float32([[1,0,100],[0,1,100]])
    dst = cv2.warpAffine(color_img, M, (dst.shape[0] * 5,dst.shape[1]*3))

    #plotting to know points of correspondence
    plt.subplot(1, 2, 1), plt.imshow(dst), plt.title('destination')
    plt.subplot(1, 2, 2), plt.imshow(src, cmap='gray'), plt.title('source')
    plt.show()
    
    #finding homography matrix
    dst_common = [[550.5, 160.6], [516, 199], [638.7, 164.6], [712, 158.7], [708, 180.5], [675, 375.5], [721, 329]]
    src_common = [[70.4, 21.4], [31.8, 61], [159.5, 35], [231.7, 34.2], [224.8, 55], [192.1, 245], [235.6, 200.5]]
    src_pts = np.asarray(src_common)
    dst_pts = np.asarray(dst_common)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    #get warped
    warped = cv2.warpPerspective(src, M, (dst.shape[1], dst.shape[0]))
    cv2.imwrite('warped.jpg', warped)

    #stich the two images
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if(np.linalg.norm(dst[i, j]) == 0):
                dst[i, j] = warped[i, j]

    cv2.imwrite('finalimage.jpg', dst)

    
    
    


    
