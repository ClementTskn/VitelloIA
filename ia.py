import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IM_SIZE = 80

# Initialisation du CNN
ia = models.Sequential()

# Couches de convolution et Max Pooling + Dropouts

ia.add(layers.Conv2D(8, (3,3),
                     activation = "relu", 
                     input_shape = (IM_SIZE, IM_SIZE, 1)))
ia.add(layers.MaxPooling2D((2,2)))
ia.add(layers.Conv2D(16, (3,3),
                     activation = "relu"))
ia.add(layers.Dropout(0.03))
ia.add(layers.MaxPooling2D((2,2)))


# Flattening
ia.add(layers.Flatten())

# Couches complètement connectées

ia.add(layers.Dense(64, activation = "relu"))
ia.add(layers.Dropout(0.3))
ia.add(layers.BatchNormalization())
ia.add(layers.Dense(64, activation = "relu"))
ia.add(layers.Dropout(0.1))
ia.add(layers.BatchNormalization())
ia.add(layers.Dense(6, activation = "softmax"))

# Compilaiton du réseau
ia.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Résumé du bordel
ia.summary()

# Importation du dataset et ajout d'images
A_cells_gen = ImageDataGenerator(
        rescale=1/.255,
        shear_range=0.2,
        horizontal_flip=False,
        vertical_flip=False,
        validation_split = 0.2)
training_dataset = A_cells_gen.flow_from_directory(
        "./photos/cellules classées/Allstades",
        target_size=(IM_SIZE,IM_SIZE),
        batch_size=32,
        subset="training",
        color_mode="grayscale",
        class_mode="sparse")

validation_dataset = A_cells_gen.flow_from_directory(
        "./photos/cellules classées/JusteClasses",
        target_size=(IM_SIZE,IM_SIZE),
        batch_size=32,
        subset="validation",
        color_mode="grayscale",
        class_mode="sparse")

history = ia.fit_generator(training_dataset,
                           steps_per_epoch = 200,
                           epochs = 5,
                           validation_data = validation_dataset,
                           validation_steps = 80)