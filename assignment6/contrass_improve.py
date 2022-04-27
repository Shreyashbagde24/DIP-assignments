from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#read image
img1 = mpimg.imread('image.jpg')

#Increased contrast
img2 = img1*2

#decreased contrast
img3 = img1/2

#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(1,3) 
axarr[0].imshow(img1)
axarr[0].set_title('original image')
axarr[1].imshow(img2)
axarr[1].set_title('Increased contrast image')
axarr[2].imshow(img3)
axarr[2].set_title('decreased contrast image')
plt.show();