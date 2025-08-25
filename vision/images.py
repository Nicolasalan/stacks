import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

image = mpimg.imread("/Users/nicolasalan/Documents/stacks/examples/image.png")

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray_image, cmap='gray')
