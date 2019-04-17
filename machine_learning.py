import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, sys

#Celle là fonctionne sans doute mieux que le pâté du dessus(et surtout,je la comprend)
def convert_image(file):
    img2 = cv2.imread(file)
    img2= cv2.resize(img2,dsize=(500,500), interpolation = cv2.INTER_CUBIC)
    gray_image = cv2.cvtColor( img2,cv2.COLOR_BGR2GRAY )
    np_image_data = np.asarray(gray_image)
    np_final = np.expand_dims(np_image_data,axis=0) #convert to float
    return np_final

#mnist= #dataset des images
    
#Reconnaissance faciale : https://github.com/msilanus/faceReco (Juste car c'est tout de même stylé)


#On convertie les images en matrices,puis on place ces matrices dans une autre matrice,créant ainsi un tenseur d'ordre 3
#On obtient ainsi une dataset sur laquelle on pourra se balader pour chopper toutes les infos nécessaires
#Il faut encore déterminer comment se balader dans le fichier pour que le programme navigue dans le dossier des images et 
#entre les données dans le tenseur

#Les images sont converties en array et sont placés dans un tenseur d'ordre 3
path = "/home/benjamin.massoteau/image_projet/"
dirs = os.listdir( path )
n=len(dirs)
data=[]
i=0
for i in range(n):
   img=convert_image(path+dirs[i])
   data.append(img)
   
print(data)
   
        
#Programme d'entraînement à adapter une fois que la dataset sera prête
"""(x_train,y_train),(x_test,y_test)=mnist.load_data() 
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
              metrics=['accuracy']) #A adapter si jamais t'arrives à comprendre les différents param' proposés

plt.imshow(image,cmap=plt.cm.binary) #Voir l'image
plt.show()"""
