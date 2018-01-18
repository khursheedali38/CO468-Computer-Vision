import numpy as np
import cv2
from matplotlib import pyplot as plt

# Applying Blur
img = cv2.imread('roof.jpg',0)
imgblur = cv2.GaussianBlur(img, (3, 3), 0)
plt.subplot(121);plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.axis('off')
plt.subplot(122);plt.imshow(imgblur, cmap = 'gray', interpolation = 'bicubic')
plt.axis('off')
plt.show()

# Applying prewit or sobel operator
kernel = np.matrix('-1 0 1; -1 0 1; -1 0 1')
H = cv2.filter2D(imgblur, -1, kernel)
V = cv2.filter2D(imgblur, -1, kernel.transpose())
Gh = np.multiply(H, H)
Gv = np.multiply(V, V)
Ghv = np.sqrt(Gh + Gv)
plt.subplot(121);plt.imshow(H, cmap='gray', interpolation = 'bicubic')
plt.axis('off')
plt.subplot(122);plt.imshow(V, cmap='gray', interpolation='bicubic')
plt.axis('off')
plt.show()
cv2.waitKey(-1)







