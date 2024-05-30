from skimage import morphology, transform, feature, measure
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from scipy import ndimage

img_test = [np.array([[256,256,256],[256,256,256]]),
            np.array([[256,256,256],[256,256,256]])]
img_test = [np.array([[256,256,256],[239,239,239]]),
            np.array([[0,0,0],[0,0,0]])]
plt.figure()
plt.imshow(img_test)
plt.show()


# Charger l'image
image = ski.io.imread('boulon_entier.jpg')
# Convertir en niveaux de gris
gray_image = ski.color.rgb2gray(image)
#/!\ Par région (watershed)
gradient = ski.filters.sobel(gray_image)

#histogramme gradiant pour marqueurs
# hist3, hist_centers3 = ski.exposure.histogram(gradient)
# fig, axes = plt.subplots(1, 2, figsize=(8, 3))
# axes[0].imshow(gradient, cmap=plt.cm.gray)
# axes[0].set_axis_off()
# axes[1].plot(hist_centers3, hist3, lw=2)
# axes[1].set_title('histogram of gray values')

# Définir les marqueurs
markers = np.zeros_like(gray_image)
foreground, background = 1, 2
seuil = 0.025 #[0;1]
markers[gradient < seuil] = background
markers[gradient > seuil] = foreground
segmentation_result = ski.segmentation.watershed(gradient, markers)
labels = measure.label(segmentation_result == foreground, background=background, connectivity=2 )

#Verif label
x, y = labels.shape
get_number = True
n = 0
is_number = 1
nb_pixel_min = 25
list_label = {}
#Supprime label inutile
while get_number == True : 
    list_i = []
    list_j = []
    n = 0 
    for i in range(y):
        for j in range(x):
            if labels[i][j] == is_number :
                n += 1 
                list_i.append(i)  
                list_j.append(j)        
    #Creer fct sup_label(seq, nb_min)                     
    if n == 0 :
        get_number = False
    elif n <= nb_pixel_min :
        for i, j in zip(list_i, list_j) :
            labels[i][j] = 0
    else : 
        list_label[is_number] = {'i_min' : min(list_i),
                                 'i_max' : max(list_i),
                                 'j_min' : min(list_j),
                                 'j_max' : max(list_j),
                                 'Nb_point' : n
                                 }
    is_number += 1
    
#Extend les labels
expanded = ski.segmentation.expand_labels(labels, distance=20)

while get_number == True : 
    list_i = []
    list_j = []
    n = 0 
    for i in range(y):
        for j in range(x):
            if labels[i][j] == is_number :
                n += 1 
                list_i.append(i)  
                list_j.append(j)

#Nouvelle photo
new_pict = []
for number_label, label in list_label.items():
    # Appliquer le masque à l'image originale
    dimension = max(label['i_max']-label['i_min'], 
                    label['j_max']-label['j_min'])
    segmented_image = np.zeros((dimension, dimension))
    for i in range(label['i_min'], label['i_max']):
        for j in range(label['j_min'], label['j_max']):
            if expanded[i][j] != number_label:
                segmented_image[i - label['i_min']][j - label['j_min']] = 1
            else:
                segmented_image[i - label['i_min']][j - label['j_min']] = gray_image[i][j]
    for i in range(dimension):
            for j in range(dimension):
                if segmented_image[i][j] == 0 :
                    segmented_image[i][j] = 1
    new_pict.append(segmented_image)
    plt.figure()
    plt.imshow(segmented_image, cmap='gray')
    plt.title(f'Segment {i+1}')
    plt.axis('off')
    plt.show()

    
# Sauvegarder la région segmentée
# output_path = os.path.join(output_dir, f'segment_{i+1}.png')
# ski.io.imsave(output_path, segmented_image)
# print(f'Segment {i+1} saved: {output_path}')
     

output_dir = './'

# Show the segmentations.
if 1 == 1 :
    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(9, 5),
        sharex=True,
        sharey=True,
    )

    axes[0].imshow(gray_image, cmap="Greys_r")
    axes[0].set_title("Original")

    axes[1].imshow(segmentation_result, cmap="nipy_spectral")
    axes[1].set_title("Sobel+Watershed")

    color1 = ski.color.label2rgb(labels, image=gray_image, bg_label=0)
    axes[2].imshow(color1)
    axes[2].set_title("Sobel+Watershed (seg)")


    color2 = ski.color.label2rgb(expanded, image=gray_image, bg_label=0)
    axes[3].imshow(color2)
    axes[3].set_title("Expanded labels (seg)")
    
plt.show()
    
region = measure.regionprops(labels)   

for i, region in enumerate(region):
    # Créer un masque pour la région
    mask = expanded == region.label
    
    # Appliquer le masque à l'image originale
    segmented_image = np.zeros_like(image)
    for c in range(3):  # Pour chaque canal de couleur
        segmented_image[..., c] = image[..., c] * mask
    
    # Sauvegarder la région segmentée
    output_path = os.path.join(output_dir, f'segment_{i+1}.png')
    ski.io.imsave(output_path, segmented_image)
    print(f'Segment {i+1} saved: {output_path}')
    
    # Afficher la région segmentée
    plt.figure()
    plt.imshow(segmented_image)
    plt.title(f'Segment {i+1}')
    plt.axis('off')
    plt.show()

#other seuillage
if 1==1:
    #/!\ Seuilage
    # Appliquer un seuillage global (Otsu)
    thresh = ski.filters.threshold_otsu(gray_image)
    binary_image = gray_image > thresh

    #/!\ Contours
    # Détection des bords avec Canny
    edges = feature.canny(gray_image, sigma=1.0)

    #/!\ Par clustering (K-means)
    # Aplatir l'image en un tableau 1D
    flat_image = gray_image.flatten().reshape(-1, 1)
    # Appliquer K-means
    kmeans = KMeans(n_clusters=2, random_state=0).fit(flat_image)
    segmented_image = kmeans.labels_.reshape(gray_image.shape)


# Afficher l'image originale et l'image avec l'arrière-plan supprimé
if 1==1:
    plt.figure(figsize=(15, 5))
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title('Image originale')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Image gris')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Par Seuillage')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(edges, cmap='gray')
    plt.title('Par Contours')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(segmentation_result, cmap='nipy_spectral')
    plt.title('Par Region')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(segmented_image, cmap='gray')
    plt.title('Par clustering')
    plt.axis('off')



