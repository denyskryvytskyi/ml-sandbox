import cv2
# import dlib
import sys
import numpy as np
from tkinter import filedialog
from matplotlib import pyplot as plt
def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

image = cv2.imread('images/1.jpg')
copy = image.copy()
# Create a mask (of zeros uint8 datatype) that is the same size (width, height) as our original image 
mask = np.zeros(image.shape[:2], np.uint8)
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
x, y , w, h = cv2.selectROI("select the area", image)
start = (x, y)
end = (x + w, y + h)
rect = (x, y , w, h)

cv2.rectangle(copy, start, end, (0,0,255), 3)
imshow("Input Image", copy)


cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 100, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
image = image * mask2[:,:,np.newaxis]
imshow("Mask", mask * 80)
imshow("Mask2", mask2 * 255)
imshow("Image", image)