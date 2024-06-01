import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

CLASS_NAMES = ['nut', 'screw']

if __name__ == '__main__':
    #Charge le modèle entrainé precedement
    model = tf.keras.models.load_model('model_reconize_screw_nut.h5')
    
    #Chemin d'accès de l'image
    img_path = '0_data_base_bin_nut.png'
    #Charge l'image à tester
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

    #PRédictions du modèle sur l'image
    probability_model = tf.keras.Sequential([model, 
                                            tf.keras.layers.Softmax()]) #Sortie en probabilité
    predictions = probability_model.predict(img_array) #Fait un prediction sur img_test

    #Affiche la prediction de l'ia
    print(CLASS_NAMES[np.argmax(predictions[0])])
    



