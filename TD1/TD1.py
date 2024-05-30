from skimage import io
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import ndimage
import os


#Verifie que l'image est bien présente dans le fichier
path = "boulon_entier.jpg"
files = os.listdir("./")
if path in files:
    print("Ok, the file is in {0}".format(files))
else:
    print("The file is not in {0} , retry !".format(files))
  
    
#Initialise et affiche l'image     
im = io.imread(path)
plt.figure()
plt.imshow(im)
plt.show()


#Affiche des infos sur l'image
print(f"The image is a {type(im)}")
print(f"The shape of this numpy array is {im.shape} and the data type is {im.dtype}")
print(f"This image has {im.size} pixels")
print(f"The height is {im.shape[0]} and the width is {im.shape[1]}")
print(f"The image has {im.shape[2]} channels")


#Plot the histogram of the image.
red = im[:,:,0]
red_f = red.flatten()
green = im[:,:,1]
green_f = green.flatten()
blue = im[:,:,2]
blue_f = blue.flatten()

plt.figure()
plt.hist(red_f, bins=256, histtype="step", label="red", color="red")
plt.hist(green_f, bins=256, histtype="step", label="green", color="green")
plt.hist(blue_f, bins=256, histtype="step", label="blue", color="blue")
plt.legend()
plt.show()


#Use thresholding to convert the image to a binary format.
seuil = 75
img_bin = np.zeros_like(red) 
rows, cols = red.shape

for i in range(rows):
  for j in range(cols):
    if red[i,j] > seuil : 
      img_bin[i,j] = 1

plt.figure()
plt.imshow(img_bin, cmap="gray")
plt.show() 


#Use erosion and dilation to clean the image if needed to isolate each coin.
img_ero = ndimage.binary_erosion(img_bin, structure=np.ones((10,10)))

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.set_title("binary")
ax1.imshow(img_bin, cmap="gray")
ax2.set_title("binary")
ax2.imshow(img_ero, cmap="gray")
plt.show()

#Numérote les differentes pieces
Bl, number = ndimage.measurements.label(img_ero)
plt.figure()
plt.imshow(np.where(Bl != 0, Bl, np.nan), cmap=cm.jet)
plt.show()

