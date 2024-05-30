from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import ndimage
import pandas as pd
import os 

#Verifie que l'image est bine présente dans le fichier
name = "HSLA_340.jpg"
workdir = "./"
files = os.listdir(workdir)
if name in files:
    print("Ok, the file is in {0}".format(files))
else:
    print("The file is not in {0} , retry !".format(files)) 
    
    
#Affiche l'image
im = np.array(Image.open(workdir + name))
fig, ax = plt.subplots()
ax.axis("off")
plt.imshow(im, cmap=cm.copper)
plt.colorbar()


#Plot the histogram of the image.
red = im[:,:,0]
red_f = red.flatten()
green = im[:,:,1]
green_f = green.flatten()
blue = im[:,:,2]
blue_f = blue.flatten()

plt.figure()
plt.hist(red_f, bins=np.arange(256), histtype="step", label="red", color="red")
plt.hist(green_f, bins=np.arange(256), histtype="step", label="green", color="green")
plt.hist(blue_f, bins=np.arange(256), histtype="step", label="blue", color="blue")
plt.legend()


#Use thresholding to convert the image to a binary format.
seuil = 200
img_bin = np.zeros_like(red) 
rows, cols = red.shape

for i in range(rows):
  for j in range(cols):
    if red[i,j] > seuil : 
      img_bin[i,j] = 1

plt.figure()
plt.imshow(img_bin, cmap="gray")


#Numérote les differentes pieces      
img_num, number = ndimage.measurements.label(img_bin)
plt.figure()
plt.imshow(np.where(img_num != 0, img_num, np.nan), cmap="viridis")
print(img_num)


#Mesure la taille des grains
'''
sizes = ndimage.sum(img_bin, img_num, range(number+1))
img_size = np.zeros_like(img_num, dtype=np.float64)
for i in range(1, number+1):
    img_size[img_num==i] = sizes[i] '''
row, col = img_num.shape

    
plt.figure()
plt.imshow(img_size)
plt.show()


#Orientation et forme des grains
obj = ndimage.measurements.find_objects(img_num)
plt.figure()
plt.imshow(np.array(img_size)[obj[2]])
'''
data = pd.DataFrame(
    columns=["area", "xg", "yg", "Ixx", "Iyy", "Ixy", "I1", "I2", "theta"]
)
for i in range(len(obj)):
    x, y = np.where(lab == i + 1)
    xg, yg = x.mean(), y.mean()
    x = x - xg
    y = y - yg
    A = len(x)
    Ixx = (y ** 2).sum()
    Iyy = (x ** 2).sum()
    Ixy = (x * y).sum()
    I = np.array([[Ixx, -Ixy], [-Ixy, Iyy]])
    eigvals, eigvecs = np.linalg.eig(I)
    eigvals = abs(eigvals)
    loc = np.argsort(eigvals)[::-1]
    d = eigvecs[loc[0]]
    d *= np.sign(d[0])
    theta = np.degrees(np.arccos(d[1]))
    eigvals = eigvals[loc]
    data.loc[i] = [A, xg, yg, Ixx, Iyy, Ixy, eigvals[0], eigvals[1], theta]
data.sort_values("area", inplace=True, ascending=False)
data["aspect_ratio"] = (data.I1 / data.I2) ** 0.5

data[["area", "theta", "aspect_ratio"]]

fig = plt.figure()
counter = 1
for i in data.index.values:
    ax = fig.add_subplot(3, 4, counter)
    z = Image.fromarray(np.array(im)[obj[i]])
    z = z.rotate(-data.loc[i, "theta"] + 90, expand=True)
    z = np.array(z)
    plt.imshow(z)
    plt.title(str(i))
    ax.axis("off")
    counter += 1
    plt.grid()

'''

#plt.show()





