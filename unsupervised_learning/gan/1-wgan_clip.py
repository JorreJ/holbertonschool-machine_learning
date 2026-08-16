#!/usr/bin/env python3
"""Module that defines a Wasserstein GAN model with weight clipping."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


class WGAN_clip(keras.Model):
    """Represent a Wasserstein GAN model with weight clipping."""

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """Initialize the WGAN_clip model.

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
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5  # standard value, but can be changed if necessary
        self.beta_2 = .9  # standard value, but can be changed if necessary

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: -tf.math.reduce_mean(
            self.discriminator(x)
        )
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss
        )

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: (
            tf.math.reduce_mean(self.discriminator(x))
            - tf.math.reduce_mean(self.discriminator(y))
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """Generate a batch of fake samples using the generator.

        Args:
            size (int, optional): The number of fake samples to generate.
                If None, uses self.batch_size.
            training (bool): Whether the model is in training mode.
                Defaults to False.

        Returns:
            tf.Tensor: A batch of fake samples generated from latent vectors.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Generate a random batch of real samples from the dataset.

        Args:
            size (int, optional): The number of real samples to retrieve.
                If None, uses self.batch_size.

        Returns:
            tf.Tensor: A batch of real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

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
