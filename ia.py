import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialisation du CNN
ia = models.Sequential()

# Couches de convolution et Max Pooling + Dropouts

ia.add(layers.Conv2D(32, (3,3),
                     activation = 'relu', 
                     input_shape = (80, 80, 1)))
ia.add(layers.MaxPooling2D((2,2)))
ia.add(layers.Dropout(0.3))
ia.add(layers.Conv2D(64, (3,3),
                     activation = 'relu'))
ia.add(layers.Dropout(0.3))
ia.add(layers.MaxPooling2D((2,2)))
ia.add(layers.Conv2D(64, (3,3),
                     activation = 'relu'))
ia.add(layers.MaxPooling2D((2,2)))

# Flattening
ia.add(layers.Flatten())

# Couches complètement connectées

ia.add(layers.Dense(256, activation = "relu"))
ia.add(layers.Dropout(0.3))
ia.add(layers.Dense(128, activation = "relu"))
ia.add(layers.Dense(64, activation = "relu"))
ia.add(layers.Dense(12, activation = "softmax"))

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
        horizontal_flip=True,
        vertical_flip=True,
        validation_split = 0.2)
training_dataset = A_cells_gen.flow_from_directory(
        "./photos/cellules classées",
        target_size=(80,80),
        batch_size=32,
        subset="training",
        color_mode="grayscale",
        class_mode="sparse")

validation_dataset = A_cells_gen.flow_from_directory(
        "./photos/cellules classées",
        target_size=(80,80),
        batch_size=32,
        subset="validation",
        color_mode="grayscale",
        class_mode="sparse")

history = ia.fit_generator(training_dataset,
                           steps_per_epoch = 2000,
                           epochs = 200,
                           validation_data = validation_dataset,
                           validation_steps = 800)