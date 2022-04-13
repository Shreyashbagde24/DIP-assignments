

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#read images
img1 = mpimg.imread('image1.jpg')
img2 = mpimg.imread('image1.jpg')

for i in range(0,400):  
      for j in range(0,1200):
            for k in range(0,3):
              img2[i+400,j+200,k] = 0
    
#obtain the final image by subtracting the original and edited image
img3= img1 - img2
#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(1,3) 
axarr[0].imshow(img1)
axarr[1].imshow(img2)
axarr[2].imshow(img3)
plt.show()