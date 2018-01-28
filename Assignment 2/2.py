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
    dst = cv2.imread('TestImages/Test 1/stich1.jpg', 1)
    src = cv2.imread('TestImages/Test 1/stich2.jpg', 1)
    Z = harris(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY))
    Z = cv2.dilate(Z,None)
    dst_temp = dst.copy()                         #image with corners marked
    dst_temp[Z>0.9999999999*Z.max()]=[0,0,255]
    

    #canvas image dst
    M = np.float32([[1,0,100],[0,1,100]])
    dst = cv2.warpAffine(dst, M, (dst.shape[0] * 5,dst.shape[1]*3))
    
    #plot to get points of correspondence
    plt.subplot(1, 3, 1), plt.imshow(dst), plt.title("destination")
    plt.subplot(1, 3, 2), plt.imshow(src), plt.title("source")
    plt.subplot(1, 3, 3), plt.imshow(dst_temp), plt.title("corners")
    plt.show()

    
    #finding homography matrix
    # dst_common = [[550.5, 160.6], [516, 199], [638.7, 164.6], [712, 158.7], [708, 180.5], [675, 375.5], [721, 329]]
    # src_common = [[70.4, 21.4], [31.8, 61], [159.5, 35], [231.7, 34.2], [224.8, 55], [192.1, 245], [235.6, 200.5]]
    dst_common = [[438.2, 219], [460.6, 291.3], [540.8, 412.5], [587.5, 330.6], [588.3, 307.3], [672.6, 331.4], [687.9, 342.6], [712.75, 303.3], [724, 321], [721.6, 349]]
    src_common = [[24.5, 120], [20.5, 200], [108.3, 336.8], [164.2, 250.33], [164.16, 234.37], [240, 258.3], [250.6, 266.3], [271.88, 235.703], [279, 254.4], [275.3, 279.1]]
    src_pts = np.asarray(src_common)
    dst_pts = np.asarray(dst_common)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    #get warped image
    warped = cv2.warpPerspective(src, M, (dst.shape[1], dst.shape[0]))
    cv2.imwrite('perspective image.jpg', warped)

    #stich the two images
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if(np.linalg.norm(dst[i, j]) == 0):
                dst[i, j] = warped[i, j]

    #stiched image
    cv2.imwrite('stiched image.jpg', dst)

    
    
    


    
