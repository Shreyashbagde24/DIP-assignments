import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(16,16), constrained_layout=False)

img_c1 = cv2.imread("input.jpg", 0)
img_c2 = np.fft.fft2(img_c1) #Compute the 2-dimensional discrete Fourier Transform
img_c3 = np.fft.fftshift(img_c2) #shift the zero-frequency component to the center of the spectrum.
img_c4 = np.fft.ifftshift(img_c3) #inv shift
img_c5 = np.fft.ifft2(img_c4) #ifft

plt.subplot(3,1,1)
plt.imshow(img_c1, "gray")
plt.title("Original Image")
plt.subplot(3,1,2)
plt.imshow(np.log(1+np.abs(img_c2)), "gray") #DFT+Logarithmic Transformation
plt.title("Spectrum")
plt.subplot(3,1,3)
plt.imshow(np.log(1+np.abs(img_c3)), "gray")
plt.title("Centered Spectrum")
plt.show()