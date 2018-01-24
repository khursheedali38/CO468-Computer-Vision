import cv2
import numpy as np
from matplotlib import pyplot as plt
from helper.utils import quantize


img = cv2.imread('building.jpg',0)
R, C = img.shape

#Smoothning and sobel
img = cv2.GaussianBlur(img,(5,5),1.4)
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()

gradient = np.hypot(sobelx, sobely)
theta = np.absolute(np.arctan2(sobely, sobelx))

#normalizing 
max = np.max(gradient)
min = np.min(gradient)
for i in range(R):
    for j in range(C):
        gradient[i, j] = (((gradient[i, j] - min)/(max-min))*255) + 0


#non-maximum supression
for i in range(R):
    for j in range(C):
        theta[i, j] = quantize(theta[i, j]) 
        
Z = np.zeros((R, C), dtype=np.int32)

for i in range(R):
    for j in range(C):
        try:
            if theta[i, j] == 0:
                if (gradient[i, j] >= gradient[i, j - 1]) and (gradient[i, j] >= gradient[i, j + 1]):
                    Z[i,j] = gradient[i,j]
            elif theta[i, j] == 90:
                if (gradient[i, j] >= gradient[i - 1, j]) and (gradient[i, j] >= gradient[i + 1, j]):
                    Z[i,j] = gradient[i,j]
            elif theta[i, j] == 135:
                if (gradient[i, j] >= gradient[i - 1, j - 1]) and (gradient[i, j] >= gradient[i + 1, j + 1]):
                    Z[i,j] = gradient[i,j]
            elif theta[i, j] == 45:
                if (gradient[i, j] >= gradient[i - 1, j + 1]) and (gradient[i, j] >= gradient[i + 1, j - 1]):
                    Z[i,j] = gradient[i,j]
        except IndexError as e:
            pass

#thresholding
            
lower_threshold = 40
upper_threshold = 60

strong_i, strong_j = np.where(Z > upper_threshold)

weak_i, weak_j = np.where((Z >= lower_threshold) & (Z <= upper_threshold))

zero_i, zero_j = np.where(Z < lower_threshold)

Z[strong_i, strong_j] = 255
Z[weak_i, weak_j] = 50
Z[zero_i, zero_j] = np.int32(0)


#edge tracking
for i in range(R):
    for j in range(C):
        if Z[i, j] == 50:
            try:
                if ((Z[i + 1, j] == 255) or (Z[i - 1, j] == 255)
                     or (Z[i, j + 1] == 255) or (Z[i, j - 1] == 255)
                     or (Z[i+1, j + 1] == 255) or (Z[i-1, j - 1] == 255)):
                    Z[i, j] = 255
                else:
                    Z[i, j] = 0
            except IndexError as e:
                pass
plt.subplot(1, 2, 1), plt.imshow(gradient , cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(Z, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()










