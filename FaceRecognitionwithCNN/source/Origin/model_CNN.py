import tensorflow
from tensorflow.keras import layers, models



model_training = models.Sequential([
    layers.Conv2D(32,(3,3), input_shape=(128,128,3),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.15),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.2),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(7, activation='softmax')

])

model_training.summary()
