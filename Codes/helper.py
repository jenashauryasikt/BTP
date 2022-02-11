import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# im.resize(size, Image.BILINEAR)


n = 144
r = 5
# tick = time.time()
image = Image.open('covid.jpg').convert('L').resize((n,n))
size = (720, 720)


out_image = image.resize(size, Image.BILINEAR)
ax1 = plt.subplot(1,2,1)
ax1.set_title("Original")
plt.imshow(image.crop((0,0,n,n)), cmap='gray', vmin=0, vmax=255)
ax2 = plt.subplot(1,2,2)
ax2.set_title("Bilinearly resolved")
plt.imshow(out_image.crop((0,0,n*r,n*r)), cmap='gray', vmin=0, vmax=255)
plt.savefig("somatic_cells_comparison_bilinear(5)_ori.png")