#!/usr/bin/env python3
"""Module that defines a Simple GAN model implementation using Keras."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """Represent a simple Generative Adversarial Network (GAN)."""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """Initialize the Simple_GAN model.

        Args:
            generator (keras.Model): The generator neural network model.
            discriminator (keras.Model): The discriminator neural network
                model.
            latent_generator (function): A function that generates latent space
                vectors given a batch size.
            real_examples (tf.Tensor or numpy.ndarray): Dataset of real
                examples used for training.
            batch_size (int): The batch size for training. Defaults to 200.
            disc_iter (int): Number of discriminator training iterations per
                generator iteration. Defaults to 2.
            learning_rate (float): The learning rate for Adam optimizers.
                Defaults to 0.005.
        """
        super().__init__()  # run the __init__ of Keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta1 = .5  # standard value, but can be changed if necessary
        self.beta2 = .9  # standard value, but can be changed if necessary

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape)
        )
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2
        )
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss
        )

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
            tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape))
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss
        )

    def get_real_sample(self):
        """Generate a random batch of real samples from the dataset.

        Returns:
            tf.Tensor: A batch of real samples of size batch_size.
        """
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:self.batch_size]
        return tf.gather(self.real_examples, random_indices)

    def get_fake_sample(self, training=False):
        """Generate a batch of fake samples using the generator.

        Args:
            training (bool): Whether the model is in training mode.
                Defaults to False.

        Returns:
            tf.Tensor: A batch of fake samples generated from latent vectors.
        """
        return self.generator(
            self.latent_generator(self.batch_size), training=training
        )

    def train_step(self, useless_argument):
        """Perform one training step for both discriminator and generator.

        Args:
            useless_argument: Dummy argument required by Keras train_step.

        Returns:
            dict: A dictionary containing 'discr_loss' and 'gen_loss'.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)
                real_preds = self.discriminator(real_sample)
                fake_preds = self.discriminator(fake_sample)
                discr_loss = self.discriminator.loss(real_preds, fake_preds)
            discr_gradient = tape.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradient, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            fake_preds = self.discriminator(fake_sample)
            gen_loss = self.generator.loss(fake_preds)
        gen_gradient = tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
