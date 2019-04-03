import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#askip la fonction permet de transformer une image en array...enfin je crois
#En tout cas,on devrait pouvoir l'utiliser pour l'apprentissage 
#lien : https://www.kaggle.com/freeman89/create-dataset-with-tensorflow

def decode_image(image_file_names, resize_func=None):
    
    images = []
    
    graph = tf.Graph()
    with graph.as_default():
        file_name = tf.placeholder(dtype=tf.string)
        file = tf.read_file(file_name)
        image = tf.image.decode_jpeg(file)
        if resize_func != None:
            image = resize_func(image)
    
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()   
        for i in range(len(image_file_names)):
            images.append(session.run(image, feed_dict={file_name: image_file_names[i]}))
            if (i+1) % 1000 == 0:
                print('Images processed: ',i+1)
        
        session.close()
    
    return images

#mnist= #dataset des images

(x_train,y_train),(x_test,y_test)=mnist.load_data() 
#Sans doute plus compliqué que ça,recherche à faire pour avoir la dataset comme il faut

x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)
x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

i=0
for i in range (3):
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    
model.add(tf.keras.layers.Dense(6, activation=tf.nn.softmax)) #6 noeuds : 1 par sortie

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
