import tensorflow as tf
from tensorflow import keras
import numpy as np

database = keras.datasets.fashion_mnist

(images_train, label_train), (images_test, label_test) = database.load_data()

images_train = images_train / 255
images_test = images_test / 255

images_train = images_train.reshape(len(images_train), 28, 28, 1)
images_test = images_test.reshape(len(images_test), 28, 28, 1)


def build_model(hp):
    model = keras.Sequential([
        keras.layers.Conv2D(
            filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
            kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
            activation='relu',
            input_shape=(28, 28, 1)
        ),
        keras.layers.Conv2D(
            filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
            kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
            activation='relu'
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(
            units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
            activation='relu'
        ),
        keras.layers.Dense(units=10, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

model_tuning = RandomSearch(build_model, objective='val_accuracy',
                            max_trials=5
                            )

model_tuning.search(images_train, label_train, epochs=3, validation_split=0.2)

best_model = model_tuning.get_best_models(num_models=1)[0]

best_model.fit(images_train, label_train, epochs=10, validation_split=0.2, initial_epoch=3)