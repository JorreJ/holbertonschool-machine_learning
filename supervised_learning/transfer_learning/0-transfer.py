#!/usr/bin/env python3
"""Module that trains a transfer learning model on CIFAR-10 using ResNet50V2."""

from tensorflow import keras as K


try:
    K.config.enable_unsafe_deserialization()
except AttributeError:
    pass


def preprocess_data(X, Y):
    """Preprocess image and label data for ResNet50V2.

    Args:
        X (numpy.ndarray): Image dataset of shape (m, 32, 32, 3) where:
            - m is the number of data points.
            - 32x32 is the height and width of each image.
            - 3 is the number of channels.
        Y (numpy.ndarray): One-hot encoding labels of shape (m, 1) or (m,).

    Returns:
        tuple:
            - X_p (numpy.ndarray): Preprocessed image dataset.
            - Y_p (numpy.ndarray): One-hot encoded labels of shape (m, 10).
    """
    X_p = K.applications.resnet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()

    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_valid_p, Y_valid_p = preprocess_data(X_valid, Y_valid)

    model = K.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(160, 160, 3)
    )
    model.trainable = False

    final_input = K.Input(shape=(32, 32, 3))

    resized = K.layers.Lambda(lambda img:
                              K.layers.Resizing(160, 160)(img))(final_input)

    model_out = model(resized, training=False)

    x = K.layers.GlobalAveragePooling2D()(model_out)
    x = K.layers.Dense(512, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    final_out = K.layers.Dense(10, activation='softmax')(x)

    cifar10_model = K.Model(inputs=final_input, outputs=final_out)
    cifar10_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    cifar10_model.fit(
        X_train_p,
        Y_train_p,
        validation_data=(X_valid_p, Y_valid_p),
        epochs=10,
        batch_size=64)

    cifar10_model.save('cifar10.h5')
