import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image1 = Image.open('LapMCM/face2_5_smooth.jpg').convert('L')
image2 = Image.open('face2_5_smooth.jpg').convert('L')

img1 = np.asarray(image1)
img2 = np.asarray(image2)

img = abs(img2-img1)

plt.imshow(img, cmap='gray')
plt.savefig('Comps/Diff_face2_5_smooth_Lap.png')
plt.show()