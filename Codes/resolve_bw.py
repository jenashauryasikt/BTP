from PIL import Image, ImageDraw
from LapMCM_regress import rbf, LapMCM_regress
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import cvxpy as cp
import time
import mosek
from sklearn.neighbors import DistanceMetric

def resolve_bw(image, m):
    
    image = image.convert('L')
    y = image.resize((m,m))
    x = np.asarray(y)
    x=np.array(x)
    l=[]
    u=[]
    e=2
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
                l.append([i, j, x[i, j]])
                l.append([i, j, x[i, j]])
                
    for i in range(50):
        for j in range(50):
            if (i%5!=0 and j%5!=0):
                u.append([float(i/5), float(j/5)])
    
    l=np.array(l)
    u=np.array(u)

    opt={
        'dataset': 'image',
        'neighbor_mode':'connectivity',
        'n_neighbor'   : 10,
        't':            1,
        'kernel_function':rbf,
        'kernel_parameters':{'gamma':0.5},
        'gamma_I':0.5,
        'gamma_A':0.0125,
        'C':100,
        'Ch':1,
        'Cb':1,
        'itr':500,
        'eta' : 0.001,
        'eps': 0.001}

    s=LapMCM_regress(opt)
    s.fit(l[:,0:2],l[:,2],u)

    X = np.mgrid[0:m:1, 0:m:1].reshape(2,-1).T
    Y_= s.decision_function(X)

    arr = np.zeros((m, m))
    ixd = 0
    err = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
                arr[i,j] = Y_[ixd]
                err += (Y_[ixd]-x[i,j])**2
                ixd += 1
    print(err)


    #lena = arr
    #xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]
    #fig = plt.figure(figsize=(8,8))
    #ax = fig.gca(projection='3d')
    #ax.plot_surface(xx, yy, arr,rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0)

    y=image
    x = np.asarray(y)
    #plt.imshow(x, cmap='gray', vmin=0, vmax=255)
    #plt.show()
    x=np.array(x)

    lena = y
    xx, yy = np.mgrid[0:lena.size[0], 0:lena.size[1]]

    # create the figure

    #fig = plt.figure(figsize=(8,8))
    #ax = fig.gca(projection='3d')
    #ax.plot_surface(xx, yy, x ,rstride=1, cstride=1, cmap=plt.cm.jet, linewidth=0)
    # show it
    #plt.show()

    X = []
    for i in range(50):
        for j in range(50):
                X.append([float(i/5), float(j/5)])
                
    X = np.array(X)
    arr = np.zeros((50, 50))
    Y_ = s.decision_function(X).reshape(-1)

    for i, y in enumerate(Y_):
        arr[int(X[i, 0]*5), int(X[i, 1]*5)] = y

    #plt.figure(figsize=(8,8))
    #plt.imshow(arr, cmap='gray')
    #lena = arr
    #xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]

    # create the figure
    #fig = plt.figure(figsize=(8,8))
    #ax = fig.gca(projection='3d')
    #ax.scatter3D(xx, yy, arr)
    # show it
    #plt.show()

    plt.figure(figsize=(12,12))
    ax1 = plt.subplot(1,2,1)
    im=image
    ax1.set_title('Original')
    x = np.asarray(im)
    plt.imshow(x, cmap='gray', vmin=0, vmax=255)
    ax2 = plt.subplot(1,2,2)
    ax2.set_title("Super-resolved")
    plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
    plt.savefig("resolved_bw.png")
    #plt.show()
    
    return arr
   

n = 48
tick = time.time()
image=Image.open('face1.jpg').convert('L').resize((n,n))
#resolve_bw(image)


cim = np.zeros((n*5, n*5))
for i in range(0, n-1, 8):
  for j in range(0, n-1, 8):
    cim[i*5:i*5+40, j*5:j*5+40] = resolve_bw(image.crop((j, i, j+8, i+8)), 8)

cim = cim-np.min(cim)
cim = (cim/np.max(cim))*255
cim = Image.fromarray(np.uint8(cim)).convert('RGB')

ax1 = plt.subplot(1,2,1)
ax1.set_title("Original")
plt.imshow(image.crop((0,0,n,n)), cmap='gray', vmin=0, vmax=255)
ax2 = plt.subplot(1,2,2)
ax2.set_title("Super-resolved")
plt.imshow(cim, cmap='gray', vmin=0, vmax=255)
plt.savefig("face1_(5)_new_comparison.png")
cim.save("face1_(5)_new_try.png")
tock = time.time()
print("Time taken :", round(tock-tick, 6))


'''
import matplotlib as mpl
dpi = mpl.rcParams['figure.dpi']
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7), dpi=dpi, sharex=True, sharey=True)
ax[0].imshow(image.crop((0,0,n,n)), cmap='gray', vmin=0, vmax=255)
ax[0].set_title("Original")
ax[1].imshow(cim, cmap='gray', vmin=0, vmax=255)
ax[1].set_title("Super-resolved")
plt.savefig("resolved_child_compare.png")
plt.show()
'''