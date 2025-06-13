# augment_module.py

import numpy as np
from sklearn.covariance import EmpiricalCovariance
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

# -------------------------
# Outlier Filtering
# -------------------------
def filter_outliers(X_new, X_ref, threshold=3.0):
    """
    Remove synthetic samples in X_new that are outliers relative to X_ref
    based on Mahalanobis distance, with mean imputation for missing values.
    """
    # Impute missing values in reference and new samples
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy='mean')
    X_ref_imp = imp.fit_transform(X_ref)
    X_new_imp = imp.transform(X_new)

    # Fit covariance on imputed reference data
    cov = EmpiricalCovariance().fit(X_ref_imp)

    # Compute Mahalanobis distances on imputed new samples
    md = cov.mahalanobis(X_new_imp)
    # Filter by threshold
    mask = md < threshold
    return X_new[mask]

# -------------------------
# MMD Loss Placeholder
# -------------------------
def mmd_loss(x, y):
    """
    Placeholder for Maximum Mean Discrepancy loss between x and y.
    """
    return tf.constant(0.0)

# -------------------------
# CVAE Implementation
# -------------------------
class CVAE(Model):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.mse = MeanSquaredError()
        # Encoder with larger hidden layers
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(latent_dim * 2),  # [mean, logvar]
        ])
        # Decoder symmetric to encoder
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(input_dim, activation=None),
        ])

    def encode(self, x):
        mean_logvar = self.encoder(x)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * logvar) * eps

    def decode(self, z):
        return self.decoder(z)

    def compute_loss(self, x, mmd_weight=1.0):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)
        recon_loss = self.mse(x, x_recon)
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
        mmd = mmd_weight * mmd_loss(x, x_recon)
        return recon_loss + kl_loss + mmd

# Increase epochs and latent dim

def augment_cvae(X, y,
                 n_samples_per_class=15000,
                 latent_dim=32,
                 epochs=400,
                 batch_size=128,
                 mmd_weight=10.0):
    """
    Train a CVAE per class with MMD regularization, generate and filter synthetic samples.
    """
    X = X.astype('float32')
    classes = np.unique(y)
    X_aug_list, y_aug_list = [], []
    for cls in classes:
        X_cls = X[y == cls]
        input_dim = X.shape[1]
        cvae = CVAE(input_dim, latent_dim)
        optimizer = Adam(1e-3)
        dataset = tf.data.Dataset.from_tensor_slices(X_cls).shuffle(1000).batch(batch_size)
        for _ in range(epochs):
            for batch in dataset:
                with tf.GradientTape() as tape:
                    loss = cvae.compute_loss(batch, mmd_weight)
                grads = tape.gradient(loss, cvae.trainable_variables)
                optimizer.apply_gradients(zip(grads, cvae.trainable_variables))
        z = tf.random.normal([n_samples_per_class, latent_dim])
        X_gen = cvae.decode(z).numpy()
        X_gen = filter_outliers(X_gen, X_cls)
        y_gen = np.full(X_gen.shape[0], cls, dtype=y.dtype)
        X_aug_list.append(X_gen)
        y_aug_list.append(y_gen)
    return np.vstack(X_aug_list), np.concatenate(y_aug_list)

# -------------------------
# GAN Implementation
# -------------------------
class GANGenerator(Model):
    def __init__(self, noise_dim=64, output_dim=None):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.InputLayer(input_shape=(noise_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(output_dim, activation=None),
        ])
    def call(self, z):
        return self.net(z)

class GANDiscriminator(Model):
    def __init__(self, input_dim):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ])
    def call(self, x):
        return self.net(x)

# Increase noise_dim and epochs

def augment_gan(X, y,
                n_samples_per_class=15000,
                noise_dim=64,
                epochs=400,
                batch_size=128,
                fm_weight=10.0):
    """
    Train a simple GAN per class with feature-matching loss, generate and filter synthetic samples.
    """
    X = X.astype('float32')
    classes = np.unique(y)
    X_aug_list, y_aug_list = [], []
    bce = BinaryCrossentropy(from_logits=False)
    for cls in classes:
        X_cls = X[y == cls]
        input_dim = X.shape[1]
        gen = GANGenerator(noise_dim, input_dim)
        disc = GANDiscriminator(input_dim)
        gen_opt = Adam(1e-4)
        disc_opt = Adam(1e-4)
        dataset = tf.data.Dataset.from_tensor_slices(X_cls).shuffle(1000).batch(batch_size)
        for _ in range(epochs):
            for real_batch in dataset:
                noise = tf.random.normal([real_batch.shape[0], noise_dim])
                with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
                    fake = gen(noise)
                    real_pred = disc(real_batch)
                    fake_pred = disc(fake)
                    d_loss = bce(tf.ones_like(real_pred), real_pred) + \
                             bce(tf.zeros_like(fake_pred), fake_pred)
                    g_adv = bce(tf.ones_like(fake_pred), fake_pred)
                    fm = tf.reduce_mean(tf.square(real_pred - fake_pred))
                    g_loss = g_adv + fm_weight * fm
                grads_d = d_tape.gradient(d_loss, disc.trainable_variables)
                disc_opt.apply_gradients(zip(grads_d, disc.trainable_variables))
                grads_g = g_tape.gradient(g_loss, gen.trainable_variables)
                gen_opt.apply_gradients(zip(grads_g, gen.trainable_variables))
        noise = tf.random.normal([n_samples_per_class, noise_dim])
        X_gen = gen(noise).numpy()
        X_gen = filter_outliers(X_gen, X_cls)
        y_gen = np.full(X_gen.shape[0], cls, dtype=y.dtype)
        X_aug_list.append(X_gen)
        y_aug_list.append(y_gen)
    return np.vstack(X_aug_list), np.concatenate(y_aug_list)
