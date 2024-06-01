from skimage import transform, measure
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
import os


class Image_ia ():
    
    def __init__(self, path:str, seuil=0.025, dimension=100, label_bg=2, label_connect=2, distance_expanse=20) -> None:
        """Initialise une photo pour pouvoir ensuite l'utilise par l'IA

        Args:
            path (str): chemin d'acces de la photo
            seuil (float, optional): Seuil entre bg et fg. Comprie entre [0;1]. Defaults to 0.025.
            label_bg (int, optional): Comprend le bg ou non dans la labelisation {None, 1, 2}. Defaults to 2.
            label_connect (int, optional): Nombre de pixel entre 2 pour les connecte ensemble. Defaults to 2.
            dimension (int, optional): Dimension de l'image final
        """
        #Caractéristique img
        self.path = path
        self.seuil = seuil
        self.dimension = dimension
        self.label_bg = label_bg
        self.label_connect = label_connect
        self.distance_expanse = distance_expanse
        
        #Differente image
        self.img_whole = ski.io.imread(path)
        self.gray_img_whole = ski.color.rgb2gray(self.img_whole)
        self.gradient_img_whole = ski.filters.sobel(self.gray_img_whole)
        self.imgs = self.__segmentation()
        self.gray_imgs = [ski.color.rgb2gray(img) for img in self.imgs]
        self.normalize_gray_imgs = [transform.resize(img, (self.dimension,self.dimension), anti_aliasing=True) for img in self.gray_imgs]
        self.normalize_gradient_imgs = [ski.filters.sobel(img) for img in self.normalize_gray_imgs]
        self.normalize_bin_imgs = [bin_image(img) for img in self.normalize_gradient_imgs]
     
                
    def __segmentation(self):
        """Divise l'image principale en fonction du nombre de pièce présente sur la photo

        Args:
            seuil (float, optional): Seuil entre bg et fg. Comprie entre [0;1]. Defaults to 0.025.
            label_bg (int, optional): Comprend le bg ou non dans la labelisation {None, 1, 2}. Defaults to 2.
            label_connect (int, optional): Nombre de pixel entre 2 pour les connecte ensemble. Defaults to 2.

        Returns:
            list : Retourn une liste des differentes images
        """
        #Segmente l'image en fonction du filtre gradient
        labels = labelisation(self.gradient_img_whole, self.seuil, self.label_bg, self.label_connect)
         
        #Supprime les labels
        labels, enum_label = revome_label(labels)
            
        #Elargie les labels
        label_expanded = ski.segmentation.expand_labels(labels, distance=self.distance_expanse)
        
        #Crée les imgages des different labels
        imgs = create_img_label(label_expanded, enum_label, self.img_whole)        
        for img in imgs : #Parcour les differente image pour supprime celle inutile (blanc/noir)
            if img.max==0 or img.min==255 :
                imgs.remove(img)
        
        #Retourne les images des différent labels
        return imgs  
    
    
def labelisation(img, seuil=0.025, label_bg=2, label_connect=2):
    """Permet de differencier les diffentes zones de chaque pièces

    Args:
        img (array): image filter avec le filtre lobel
        seuil (float, optional): Seuil entre bg et fg. Comprie entre [0;1]. Defaults to 0.025.
        label_bg (int, optional): Comprend le bg ou non dans la labelisation {None, 1, 2}. Defaults to 2.
        label_connect (int, optional): Nombre de pixel entre 2 pour les connecte ensemble. Defaults to 2.

    Returns:
        array : Retourne une photo avec les different label
    """
    #Crée un image de taille identique 
    markers = np.zeros_like(img)
    
    #Identifie le foreground et le background grace au seuil choisie
    foreground, background = 1, 2
    markers[img < seuil] = background
    markers[img > seuil] = foreground
    
    #Applique la segmentation
    segmentation_result = ski.segmentation.watershed(img, markers)
    
    #Repère les différent labels
    labels = measure.label(segmentation_result == foreground, background=label_bg, connectivity=label_connect)
    
    #Retourne une photo des labels
    return labels    
           
    
def revome_label(labels, pixel_min=1000):
        """Supprime les labels trop petit

        Args:
            labels (array): image des different label
            pixel_min (int, optional): nombre minimal de pixel requis pour garder un label. Defaults to 1000.
        Returns:
            (array, dict) : Retourne une photo avec les labels supprimer et une dictionnaire avec leur caractérisitque
        """
        #Basic caract
        enum_label = {} #Stock les caractéristique des labels
        x, y = labels.shape #Dimensions de la photo
        get_number = True #Verifie que le label existe
        n = 0 #Nombre de point
        is_number = 1 #Defini le label tester
        
        #Boucle parcourant les labels un à un
        while get_number == True : 
            list_i = [] #Liste des pixels en i
            list_j = [] #Liste des pixels en j
            n = 0       #Nombre de pixel
            
            for i in range(x): #Boucle parcourant l'image des labels
                for j in range(y):
                    if labels[i][j] == is_number : #Verifie que le pixel appartient au label
                        #Ajoute le pixel à la liste
                        n += 1 
                        list_i.append(i)  
                        list_j.append(j)  
                  
            if n == 0 : #Si 0 point detecter alors il n'y a plus de labels on ferme la boucle
                get_number = False
            elif n <= pixel_min : #Si moins de pixel que voulue alors on sup le label
                for i, j in zip(list_i, list_j) :
                    labels[i][j] = 0
            else : #Sinon on stock les caractériqtique du label
                enum_label[is_number] = {'i_min' : min(list_i),
                                        'i_max' : max(list_i),
                                        'j_min' : min(list_j),
                                        'j_max' : max(list_j),
                                        'nb_point' : n
                                        }
            
            #Identation de 1 pour tester le prochain label    
            is_number += 1  
        
        return labels, enum_label #Return les labels restant et leurs caractéristique

        
def create_img_label(labels, enum_label, image):
    """Crée une nouvelle image pour chaque label et renvoie une liste de celles-ci

        Args:
            enum_label (dict): caractéristique de chaque label
            label_expanded (array): photo des labels voulues

        Returns:
            list : Retourne une liste des nouvelle photo
        """
    #Creer la list qui contiendra les futures images
    new_pict = []
    
    #Boucle pour le nombre de labels (et donc de photos)
    for number_label, label in enum_label.items(): 
        
        dimension = max(label['i_max']-label['i_min'], #Prendre les dimensions de la futur image
                        label['j_max']-label['j_min'])
        segmented_image = np.zeros((dimension, dimension, 3), dtype=('uint8')) #Creer une image noir au dimensions

        #Remplace l'image noir par celle voulue
        for i in range(label['i_min'], label['i_max']): #Parcoure l'emplacement de l'image danc celle de base
            for j in range(label['j_min'], label['j_max']):
                if labels[i][j] != number_label: #Si pixel n'appartient pas au label alors on le met noir
                    segmented_image[i - label['i_min']][j - label['j_min']] = np.array([0,0,0], dtype=('uint8'))
                else: #Sinon il prend la valeur de l'image initial
                    segmented_image[i - label['i_min']][j - label['j_min']] = image[i][j]
        
        #Remplace les pixels noir pour obtenir un background uniforme
        for i in range(dimension):
            for j in range(dimension):
                if np.array_equal(segmented_image[i][j],np.array([0,0,0]), equal_nan=False) :
                    if i == 0 :
                        segmented_image[i][j] = segmented_image[i][j-1]
                    elif j == 0 :
                        segmented_image[i][j] = segmented_image[i-1][j]
                    else :
                        segmented_image[i][j] = segmented_image[i-1][j-1] 
                        
        #Ajoute l'image à la liste d'images
        new_pict.append(segmented_image) 
        
    #Retourne la liste d'image   
    return new_pict
    

def bin_image(img, seuil=0.025):
    """Binarise une image

    Args:
        img (array): image à binarisé
        seuil (float, optional): Seuil de binarisation [0;1]. Defaults to 0.025.

    Returns:
        array : Retourne une image binarisé
    """
    #Crée un image de taille identique
    markers = np.zeros_like(img)
    
    #Identifie le foreground et le background grace au seuil choisie
    foreground, background = 1, 2
    markers[img < seuil] = background
    markers[img > seuil] = foreground
    
    #Applique la binarisation
    img_watershed = ski.segmentation.watershed(img, markers)
    
    #Parcoure l'image
    for i in range(img_watershed.shape[0]):
        for j in range(img_watershed.shape[0]):
            if img_watershed[i][j] == 2: #Si le pixel appartient au foregoroung alors on le met en noir
                img_watershed[i][j] = 1.
            else:
                img_watershed[i][j] = 0. #Si le pixel appartient au background alors on le met en blanc    
    
    #Retourne l'image binarisé
    return img_watershed 
  
    
def list_files_recursively(directory):
    """Renvoie une liste des fichier present dans l'arboraisance
    à partir du chemin d'acces
    """
    list_path = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            list_path.append(os.path.join(root, file))
    return list_path


def save_normalize_bin_img(path, dimension):
    """Enregistre toute les photos dans une dimension à partir 
        des fichier image trouvé dans l'arboransance
    """

    #Liste de tout les chemin d'acces vers une image       
    liste = list_files_recursively(path)
    
    nut = 1
    screw = 1
    os.mkdir(f'data_base_bin_{dimension}')
    os.mkdir(f'data_base_bin_{dimension}/screws')
    os.mkdir(f'data_base_bin_{dimension}/nuts')
    
    for path in liste :
        img = Image_ia(path, dimension=dimension)
        for bin in img.normalize_bin_imgs:
            # Sauvegarder l'image binaire
            if "screws" in path : 
                name = f'data_base_bin_{dimension}/screws/{screw}_img_bin_screw_{dimension}.png'
                ski.io.imsave(name, (bin*255).astype('uint8'))
                screw += 1
            else :
                name = f'data_base_bin_{dimension}/nuts/{nut}_img_bin_nut_{dimension}.png'
                ski.io.imsave(name, (bin*255).astype('uint8'))
                nut += 1
            print(name)


if __name__ == "__main__":
    
    path = 'boulon_entier.jpg' #Chemin d'acces de l'image
    img = Image_ia(path) 
    
    #Affiche toute les état de l'image
    plt.figure()
    plt.title('image entière (couleur)')
    plt.imshow(img.img_whole)
    plt.show()
    
    plt.figure()
    plt.title('image entière (gris)')
    plt.imshow(img.gray_img_whole, cmap='gray')
    plt.show()
    
    plt.figure()
    plt.title('image entière (gradient)')
    plt.imshow(img.gradient_img_whole, cmap='gray')
    plt.show()
    
    plt.figure()
    plt.title('image segmenté (couleur)')
    plt.imshow(img.imgs[0])
    plt.show()
    
    plt.figure()
    plt.title('image segmenté (gris)')
    plt.imshow(img.gray_imgs[0], cmap='gray')
    plt.show()
    
    plt.figure()
    plt.title('image segmenté normalisé (gris)')
    plt.imshow(img.normalize_gray_imgs[0], cmap='gray')
    plt.show()
    
    plt.figure()
    plt.title('image segmenté normalisé (gradient)')
    plt.imshow(img.normalize_gradient_imgs[0], cmap='gray')
    plt.show()
    
    plt.figure()
    plt.title('image segmenté normalisé (binaire)')
    plt.imshow(img.normalize_bin_imgs[0], cmap='gray')
    plt.show()

    #Enregistre toute les images dans une dimension 
    path = 'data_base' #Chemin d'acces des images
    save_normalize_bin_img(path, 224) #Sauvegarde les images dans un dossier
                


            
    
    
    