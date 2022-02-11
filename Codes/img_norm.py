import cv2
import PIL
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('TFMCM/face1_(5)_new_try.png', 0)
m, n = img.shape
img_new = np.zeros([m, n])

for i in range(m):
	if (i%40==0):
		try:
			img_new[i-1,:] = 255
			img_new[i,:] = 255
			img_new[i+1,:] = 255
		except:
			print("exception occurred")

for j in range(n):
	if (j%40==0):
		try:
			img_new[:,j-1] = 255
			img_new[:,j] = 255
			img_new[:,j+1] = 255
		except:
			print("exception occurred")


img_new = np.asarray(img_new, dtype='uint8')

# plt.imshow(img_new, cmap='gray')
# plt.show()

dst = cv2.inpaint(img,img_new,40,cv2.INPAINT_NS)
# cv2.imshow('dst',dst)

# plt.imshow(dst, cmap='gray')
# plt.show()

median = cv2.GaussianBlur(dst, (5,5), 0)
# # print(median)
plt.imshow(median, cmap='gray')
plt.savefig('face1_5_smooth_TF.jpg')
plt.show()