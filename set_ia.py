import tensorflow as tf

CLASS_NAMES = ['nut', 'screw'] #Liste des possibilité
dimension = 224 #Dimension de l'image
batch_size = 32 
path = f'data_base_color_{dimension}' #Chemin d'acces à la base de donnée



#Charge les imgs d'entrainement
train_ds = tf.keras.utils.image_dataset_from_directory(
  path,
  validation_split = 0.2,
  subset = "training",
  seed = 123,
  color_mode = 'rgb',
  image_size = (dimension, dimension),
  batch_size = batch_size
)

#Charge les imgs de test
test_ds = tf.keras.utils.image_dataset_from_directory(
  path,
  validation_split = 0.2,
  subset = "validation",
  seed = 123,
  color_mode = 'rgb', #'grayscale'
  image_size = (dimension, dimension),
  batch_size = batch_size
)

#Permet d'extraire les donnes plus facilement et donc évite un blocage
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Crée le model
model = tf.keras.Sequential([ #Ajoute les couche du model
    tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(dimension, dimension, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(2) #2 sortie pour les 2 possibilité
])

#Compile le model
model.compile(
    optimizer = 'adam', #Optimisateur du model 
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),   #Permet de mesure la precision du model
    metrics = ['accuracy']   #utilisées pour surveiller les étapes de formation et de test
)

#Affiche un sommaire du model
model.summary()


#Entraine le model
history = model.fit(
            train_ds, #Donnée d'entrainement
            validation_data = val_ds, #Donnée de test
            epochs = 15 #Nombre de période   
)

#Sauvegarde le model
model.save('model_reconize_screw_nut.h5')
