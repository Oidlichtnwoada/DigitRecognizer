from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf


def get_model():
    tf.keras.backend.set_floatx('float64')
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(784, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


def get_datasets():
    mnist = tf.keras.datasets.mnist
    (images_train, labels_train), (images_test, labels_test) = mnist.load_data()
    images_train, images_test = images_train / 255.0, images_test / 255.0
    images_train = images_train[..., tf.newaxis]
    images_test = images_test[..., tf.newaxis]
    train_dataset = tf.data.Dataset.from_tensor_slices((images_train, labels_train)).shuffle(10000).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((images_test, labels_test)).batch(64)
    return train_dataset, test_dataset


def train(epochs, dataset, model):
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    for epoch in range(epochs):
        for (batch, (images, labels)) in enumerate(dataset):
            train_step(images, labels, model, optimizer, loss_object)
            # print(f'batch {batch + 1} of {max(enumerate(dataset))[0] + 1} in epoch {epoch + 1} of {epochs} trained')


def train_step(images, labels, model, optimizer, loss_object):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = loss_object(labels, logits)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def test(dataset, model):
    accuracy = 0
    dataset_size = max(enumerate(dataset))[0] + 1
    for (batch, (images, labels)) in enumerate(dataset):
        predicted_labels = np.argmax(model.predict(np.squeeze(images)), axis=1)
        accuracy += np.mean(labels == predicted_labels)
        # print(f'batch {batch + 1} of {dataset_size} tested')
    return accuracy / dataset_size


mnist_model = get_model()
mnist_model.summary()
train_ds, test_ds = get_datasets()
train(16, train_ds, mnist_model)
print(f'accuracy is {test(test_ds, mnist_model)}')
