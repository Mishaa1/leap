"""
Defines a Progressively Growing GAN
"""
import os
import time
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LeakyReLU, Reshape
from tensorflow.keras.layers import UpSampling2D, AveragePooling2D, Flatten
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from progan import dataset
from progan.custom_layers import PixelNorm, MinibatchStd, FadeIn, Conv2D, Dense, wasserstein_loss

class ProgressiveGAN():
    """
    ProGAN based on implementation from
    https://github.com/PacktPublishing/Hands-On-Image-Generation-with-TensorFlow-2.0/blob/master/Chapter07/ch7_progressive_gan.ipynb
    """
    def __init__(self, results_dir, z_dim=512, resolution=512, load_path=None, start_log2_res=2, checkpoint_suffix=None,
                 learning_rate=1e-3, beta_1=0.0, beta_2=0.99, epsilon=1e-8):
        """
        Initializes ProGAN class

        Args:
            results_dir (string):                 path to directory where results are to be stored
            z_dim (int, optional):                latent noise vector size. Defaults to 512.
            resolution (int, optional):           target resolution in pixels. Defaults to 512.
            load_path (string, optional):         if not none, will load a model checkpoint at path. Defaults to None.
            start_log2_res (int, optional):       starting image resolution on log2 scale. Defaults to 2.
            checkpoint_suffix (string, optional): optional suffix to add to checkpoint name,
                                                  used to differentiate models from different sites. Defaults to None.
            learning_rate (float, optional):      optimizer learning rate, see Adam Optimizer. Defaults to 1e-3.
            beta_1 (float, optional):             optimizer beta_1, see Adam Optimizer. Defaults to 0.0.
            beta_2 (float, optional):             optimizer beta_2, see Adam Optimizer. Defaults to 0.99.
            epsilon (float, optional):           optimizer epsilon, see Adam Optimizer. Defaults to 1e-8.
        """
        self.results_dir = results_dir
        self.start_log2_res = start_log2_res
        self.resolution = resolution
        self.log2_resolution = int(np.log2(resolution))
        self.log2_res_to_filter_size = {
            0: 512,
            1: 512,
            2: 512, # 4x4
            3: 512, # 8x8
            4: 512, # 16x16
            5: 512, # 32x32
            6: 256, # 64x64
            7: 128, # 128x128
            8: 64,  # 256x256
            9: 32,  # 512x512
            10: 16  # 1024x1024
        }

        self.z_dim = z_dim
        self.alpha = np.array([[1]])
        self.initializer = tf.keras.initializers.RandomNormal(0., 1.)
        self.opt_init = {'learning_rate':learning_rate, 'beta_1':beta_1, 'beta_2':beta_2, 'epsilon':epsilon}

        self.g_loss = 0.
        self.d_loss = 0.
        self.build_all_generators()
        self.build_all_discriminators()

        if load_path:
            self.load_checkpoint(load_path)

        # initialize generator with the base
        self.val_z = tf.random.normal((12, self.z_dim))
        dummy_alpha = Input(shape=(1), name='DummyAlpha')
        rgb = self.to_rgb[2](self.generator_blocks[2].output)
        self.generator = Model([self.generator_blocks[2].input, dummy_alpha], rgb)

        # build base discriminator
        input_image = Input(shape=(4, 4, 3))
        alpha = Input(shape=(1))
        x = self.from_rgb[2](input_image)
        pred = self.discriminator_blocks[2](x)

        self.discriminator = Model([input_image, alpha], pred, name='discriminator_4x4')
        self.optimizer_discriminator = Adam(**self.opt_init)
        self.optimizer_generator = Adam(**self.opt_init)
        self.discriminator.trainable = False

        # build composite model
        pred = self.discriminator([self.generator.output, self.generator.input[1]])

        self.model = Model(inputs=self.generator.input, outputs=pred)

        if checkpoint_suffix:
            self.checkpoint_path = f"{self.results_dir}checkpoints_{checkpoint_suffix}/"
        else:
            self.checkpoint_path = f"{self.results_dir}checkpoints/"

    def load_checkpoint(self, path, log_2_res=2):
        """
        Loads a pre-saved model

        Args:
            path (str): path to checkpoint directory
        """
        for i in range(2, log_2_res+1):
            self.to_rgb[i] = load_model(f'{path}/to_rgb_{i}')
            self.from_rgb[i]= load_model(f'{path}/from_rgb_{i}')
            self.discriminator_blocks[i]= load_model(f'{path}/d_{i}')
            self.generator_blocks[i]= load_model(f'{path}/g_{i}')

        if log_2_res != 2:
            self.grow_model(log_2_res)

    def build_all_generators(self):
        """
        Builds a dictionary of all generator models to use at each resolution
        """
        # build all the generator block
        self.to_rgb = {}
        self.generator_blocks = {}
        self.generator_blocks[2] = self.build_generator_base(self.z_dim)
        self.to_rgb[2] = self.build_to_rgb(4, self.log2_res_to_filter_size[2])

        for log2_res in range(3, self.log2_resolution+1):
            res = 2**log2_res
            filter_n = self.log2_res_to_filter_size[log2_res]
            self.to_rgb[log2_res] = self.build_to_rgb(res, filter_n)

            input_shape = self.generator_blocks[log2_res-1].output[0].shape
            gen_block = self.build_generator_block(log2_res, input_shape)
            self.generator_blocks[log2_res] = gen_block

    def build_generator_base(self, input_shape):
        """
        Builds the base (first) layer of the generator model

        Args:
            input_shape (tuple): model input shape, based on latent input size
        """
        input_tensor = Input(shape=input_shape)
        x = PixelNorm()(input_tensor)
        x = Dense(8192, gain=1./8)(x)
        x = Reshape((4, 4, 512))(x)
        x = LeakyReLU(0.2)(x)
        x = PixelNorm()(x)
        x = Conv2D(512, 3, name='gen_4x4_conv1')(x)
        x = LeakyReLU(0.2)(x)
        x = PixelNorm()(x)

        return Model(input_tensor, x, name='generator_base')

    def build_generator_block(self, log2_res, input_shape):
        """
        Builds a single generator block at a given resolution

        Args:
            log2_res (int): resolution to build block for on log2 scale
            input_shape (tuple): model input shape, based on latent input size
        """
        res = 2**log2_res
        res_name = f'{res}x{res}'
        filter_n = self.log2_res_to_filter_size[log2_res]

        input_tensor = Input(shape=input_shape)
        x = UpSampling2D((2, 2))(input_tensor)

        x = Conv2D(filter_n, 3, name=f'gen_{res_name}_conv1')(x)
        x = LeakyReLU(0.2)(x)
        x = PixelNorm()(x)

        x = Conv2D(filter_n, 3, name=f'gen_{res_name}_conv2')(x)
        x = LeakyReLU(0.2)(x)
        x = PixelNorm()(x)

        return Model(input_tensor, x, name=f'genblock_{res}_x_{res}')

    def build_discriminator_base(self, input_shape):
        """
        Builds the base (last) layer of the discriminator model

        Args:
            input_shape (tuple): model input shape, based on start resolution
        """
        input_tensor = Input(shape=input_shape)

        x = MinibatchStd()(input_tensor)
        x = Conv2D(512, 3, name='gen_4x4_conv1')(x)
        x = LeakyReLU(0.2)(x)
        x = Flatten()(x)

        x = Dense(512, name='gen_4x4_dense1')(x)
        x = LeakyReLU(0.2)(x)

        x = Dense(1, name='gen_4x4_dense2')(x)

        return Model(input_tensor, x, name='discriminator_base')

    def build_discriminator_block(self, log2_res, input_shape):
        """
        Builds a single discriminator block at a given resolution

        Args:
            log2_res (int): resolution to build block for on log2 scale
            input_shape (tuple): model input shape, based resolution
        """
        filter_n = self.log2_res_to_filter_size[log2_res]
        input_tensor = Input(shape=input_shape)

        # First conv
        x = Conv2D(filter_n, 3)(input_tensor)
        x = LeakyReLU(0.2)(x)

        # Second conv + downsample
        filter_n = self.log2_res_to_filter_size[log2_res-1]
        x = Conv2D(filter_n, 3,)(x)
        x = LeakyReLU(0.2)(x)
        x = AveragePooling2D((2, 2))(x)

        res = 2**log2_res
        return Model(input_tensor, x, name=f'disc_block_{res}_x_{res}')

    def build_to_rgb(self, res, filter_n):
        """
        Layer for converting generator output to rgb

        Args:
            res (int): target resolution, should be equal to generator block resolution
            filter_n (int): filter size on last generator block
        """
        return Sequential([Input(shape=(res, res, filter_n)),
                           Conv2D(3, 1, gain=1, activation=None,
                                  name=f'to_rgb_{res}x{res}_conv')],
                          name=f'to_rgb_{res}x{res}')

    def build_from_rgb(self, res, filter_n):
        """
        Layer for decoding rgp input to discriminator

        Args:
            res (int): input resolution, should be equal to discriminator block resolution
            filter_n (int): filter size to use on convolutional layer
        """
        return Sequential([Input(shape=(res, res, 3)),
                           Conv2D(filter_n, 1, name=f'from_rgb_{res}x{res}_conv'),
                           LeakyReLU(0.2)], name=f'from_rgb_{res}x{res}')

    def build_all_discriminators(self):
        """
        Builds a dictionary of all discriminator models to use at each resolution
        """
        self.from_rgb = {}
        self.discriminator_blocks = {}

        # all but the final block
        for log2_res in range(self.log2_resolution, 1, -1):
            res = 2**log2_res
            filter_n = self.log2_res_to_filter_size[log2_res]
            self.from_rgb[log2_res] = self.build_from_rgb(res, filter_n)

            input_shape = (res, res, filter_n)
            self.discriminator_blocks[log2_res] = self.build_discriminator_block(log2_res, input_shape)

        # last block at 4x4 resolution
        log2_res = 2
        filter_n = self.log2_res_to_filter_size[log2_res]
        self.from_rgb[log2_res] = self.build_from_rgb(4, filter_n)
        res = 2**log2_res
        input_shape = (res, res, filter_n)
        self.discriminator_blocks[log2_res] = self.build_discriminator_base(input_shape)

    def grow_generator(self, log2_res):
        """
        Grows generator model to target resolution

        Args:
            log2_res (int): target resolution on log2 scale
        """
        res = 2**log2_res
        alpha = Input(shape=(1), name=f'alpha_{res}')

        x = self.generator_blocks[2].input

        for i in range(2, log2_res):
            x = self.generator_blocks[i](x)

        old_rgb = self.to_rgb[log2_res-1](x)
        old_rgb = UpSampling2D((2, 2))(old_rgb)

        x = self.generator_blocks[log2_res](x)
        new_rgb = self.to_rgb[log2_res](x)
        rgb = FadeIn()(alpha, new_rgb, old_rgb)
        self.generator = Model([self.generator_blocks[2].input, alpha], rgb,
                               name=f'generator_{res}_x_{res}')

        self.optimizer_generator = Adam(**self.opt_init)

    def grow_discriminator(self, log2_res):
        """
        Grows discriminator model to target resolution

        Args:
            log2_res (int): target resolution on log2 scale
        """
        res = 2**log2_res

        input_image = Input(shape=(res, res, 3))
        alpha = Input(shape=(1))

        x = self.from_rgb[log2_res](input_image)
        x = self.discriminator_blocks[log2_res](x)

        downsized_image = AveragePooling2D((2, 2))(input_image)
        y = self.from_rgb[log2_res-1](downsized_image)

        x = FadeIn()(alpha, x, y)
        for i in range(log2_res-1, 1, -1):
            x = self.discriminator_blocks[i](x)

        self.discriminator = Model([input_image, alpha], x,
                                   name=f'discriminator_{res}_x_{res}')

        self.optimizer_discriminator = Adam(**self.opt_init)

    def grow_model(self, log2_res):
        """
        Grow GAN to target resolution and recompile model

        Args:
            log2_res (int): target resolution on log2 scale
        """
        tf.keras.backend.clear_session()
        res = 2**log2_res
        print(f"Growing model to {res}x{res}")

        self.grow_generator(log2_res)
        self.grow_discriminator(log2_res)

        self.discriminator.trainable = False

        latent_input = Input(shape=(self.z_dim))
        alpha_input = Input(shape=(1))
        fake_image = self.generator([latent_input, alpha_input])
        pred = self.discriminator([fake_image, alpha_input])

        self.model = Model(inputs=[latent_input, alpha_input], outputs=pred)
        # self.model.compile(loss=wasserstein_loss, optimizer=Adam(**self.opt_init))

    def train_discriminator_wgan_gp(self, real_images, alpha):
        """
        Run a train step on discriminator using wassertein loss

        Args:
            real_images (tensor): real image batch used for training
            alpha (tensor): fade-in factor
        """
        batch_size = real_images.shape[0]
        real_labels = tf.ones(batch_size)
        fake_labels = -tf.ones(batch_size)

        z = tf.random.normal((batch_size, self.z_dim))
        fake_images = self.generator([z, alpha])

        with tf.GradientTape() as gradient_tape, tf.GradientTape() as total_tape:

            # forward pass
            pred_fake = self.discriminator([fake_images, alpha])
            pred_real = self.discriminator([real_images, alpha])

            epsilon = tf.random.uniform((batch_size, 1, 1, 1))
            interpolates = epsilon*real_images + (1-epsilon)*fake_images
            gradient_tape.watch(interpolates)
            pred_fake_grad = self.discriminator([interpolates, alpha])

            # calculate losses
            loss_fake = wasserstein_loss(fake_labels, pred_fake)
            loss_real = wasserstein_loss(real_labels, pred_real)
            loss_fake_grad = wasserstein_loss(fake_labels, pred_fake_grad)

            # gradient penalty
            gradients_fake = gradient_tape.gradient(loss_fake_grad, [interpolates])
            gradient_penalty = self.gradient_loss(gradients_fake)

            # drift loss
            all_pred = tf.concat([pred_fake, pred_real], axis=0)
            drift_loss = 0.001 * tf.reduce_mean(all_pred**2)

            total_loss = loss_fake + loss_real + gradient_penalty + drift_loss

            # apply gradients
            gradients = total_tape.gradient(total_loss, self.discriminator.variables)

            self.optimizer_discriminator.apply_gradients(zip(gradients, self.discriminator.variables))

        return total_loss

    def train_generator_priv(self, z, alpha, real_labels, p_disc, dp_labels, priv_ratio):
        """
        Run a train step on generator used in context of FELICIA
        Takes into account additional loss penalty from privacy generator

        Args:
            z (tesnor): latent noise
            alpha (tensor): fade-in factor
            real_labels (tensor): labels for real images for discriminator
            p_disc (PrivacyDiscriminator): model used for enforcing privacy penalty
            dp_labels (tensor): "real" labels for privacy discriminator. In this context,
                                labels should be a random set of values in the range of num_clients,
                                but doesn't include current client being trained
            priv_ratio (float): lambda used to weigh privacy penalty
        """
        with tf.GradientTape() as gradient_tape:
            # forward pass
            fake_images = self.generator([z, alpha])
            gradient_tape.watch(fake_images)

            batch_size = fake_images.shape[0]
            d_preds = self.discriminator([fake_images, alpha])
            real_labels = tf.ones(batch_size)

            dp_preds = p_disc.predict(fake_images, alpha)

            # calculate losses
            loss_disc = wasserstein_loss(real_labels, d_preds)
            total_loss = loss_disc

            loss_pdisc = 0
            if dp_labels is not None:
                loss_pdisc = tf.reduce_mean(sparse_categorical_crossentropy(dp_labels, dp_preds, from_logits=False))

            total_loss = loss_disc + priv_ratio*loss_pdisc

            # apply gradients
            gradients = gradient_tape.gradient(total_loss, self.generator.variables)
            self.optimizer_generator.apply_gradients(zip(gradients, self.generator.variables))

        return total_loss, loss_disc, loss_pdisc

    def train_step(self, data_gen, alpha):
        """
        Single train step for use in general case

        Args:
            data_gen (ImageDataGenerator): generator used to fetch training data
            alpha (tensor): fade-in factor
        """
        real_images = next(data_gen)
        self.d_loss = self.train_discriminator_wgan_gp(real_images, alpha)

        real_images = next(data_gen)
        batch_size = real_images.shape[0]
        real_labels = tf.ones(batch_size)
        z = tf.random.normal((batch_size, self.z_dim))
        alpha = tf.convert_to_tensor([alpha]*batch_size)
        self.g_loss = self.model.train_on_batch([z, alpha], real_labels)

    def train_step_felicia(self, real_images, alpha, p_disc, dp_labels, priv_ratio):
        """
        Single train step for use when using FELICIA

        Args:
            real_images (tensor): image data used for training
            alpha (tensor): fade-in factor
            p_disc (PrivacyDiscriminator): model used for enforcing privacy penalty
            dp_labels (tensor): "real" labels for privacy discriminator. In this context,
                                labels should be a random set of values in the range of num_clients,
                                but doesn't include current client being trained
            priv_ratio (float): lambda used to weigh privacy penalty
        """
        batch_size = real_images.shape[0]
        real_labels = tf.ones(batch_size)
        z = tf.random.normal((batch_size, self.z_dim))
        alpha = tf.convert_to_tensor([alpha]*batch_size)

        self.d_loss = self.train_discriminator_wgan_gp(real_images, alpha)
        self.g_loss, loss_disc, loss_pdisc = self.train_generator_priv(z, alpha, real_labels, p_disc, dp_labels, priv_ratio)

        return self.g_loss, loss_disc, loss_pdisc

    def generate(self, z):
        """
        Generate samples using generator

        Args:
            z (tensor): Latent noise with shape (batchsize, latentNoiseDimension)
        """
        images = self.generator([z, self.alpha])
        images = np.clip((images*0.5 + 0.5)*255, 0, 255)
        return images.astype(np.uint8)

    def checkpoint(self, state, log2_res, step, z=None):
        """
        Save a checkpoint of current model (both generator and discriminator)
        And save some sample images

        Args:
            log2_res (int): current model resolution on log2 scale
            step (int): step number in traninig, used to differentiate checkpoints at different training phases
            z (tensor, optional): latent noise vector, can predefine to generate similar images at different resolutions. Defaults to None.
        """
        if z is None:
            z = self.val_z

        res = 2**log2_res
        prefix = f'state_{state}_res_{res}x{res}_{step}'

        ckpt_save_path = f'{self.checkpoint_path}{prefix}'

        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)

        print('Saving checkpoint', ckpt_save_path)

        for i in range(2, log2_res+1):
            self.to_rgb[i].save(f'{ckpt_save_path}/to_rgb_{i}')
            self.from_rgb[i].save(f'{ckpt_save_path}/from_rgb_{i}')
            self.discriminator_blocks[i].save(f'{ckpt_save_path}/d_{i}')
            self.generator_blocks[i].save(f'{ckpt_save_path}/g_{i}')

        if not hasattr(self, 'alpha'):
            self.alpha = np.array([[1]])

        images = self.generate(z)
        dataset.plot_images(images, log2_res, f"{self.results_dir}saved_images/{prefix}.jpg")

    def gradient_loss(self, grad):
        """
        Calculates loss from gradient
        """
        loss = tf.square(grad)
        loss = tf.reduce_sum(loss, axis=np.arange(1, len(loss.shape)))
        loss = tf.sqrt(loss)
        loss = tf.reduce_mean(tf.square(loss - 1))
        self.penalty_const = 10
        loss = self.penalty_const * loss
        return loss

    def train(self, datasets, train_step_ratio, steps_per_phase=1000, tick_interval=500):
        """
        Full training on proGAN model

        Args:
            datasets (ImageDataGenerator): generator for training data
            train_step_ratio (dict): normalizing factor for number of steps, accounts for different batch sizes at different resolutions
            steps_per_phase (int, optional): number of steps for each resolution. Defaults to 1000.
            tick_interval (int, optional): step interval to checkpoint model at. Defaults to 500.
        """
        self.val_z = tf.random.normal((12, self.z_dim))

        for log2_res in range(self.start_log2_res, self.log2_resolution+1, 1):
            start_time = time.time()
            self.current_log2_res = log2_res

            res = 2**log2_res
            data_gen = iter(datasets[log2_res])
            print(f"Resolution {res}x{res}")

            for state in ['TRANSITION', 'STABLE']:
                if state == 'TRANSITION' and log2_res == 2:
                    continue

                steps = int(train_step_ratio[log2_res] * steps_per_phase)
                interval = int(train_step_ratio[log2_res] * tick_interval)
                for step in tqdm(range(0, steps)):
                    alpha = step/steps if state == 'TRANSITION' else 1.
                    self.alpha = np.array([[alpha]])

                    if step%interval == 0:
                        print('alpha', self.alpha)
                        elapsed = time.time() - start_time
                        start_time = time.time()
                        minutes = int(elapsed//60)
                        seconds = int(elapsed%60)
                        print(f"elapsed {minutes} min {seconds} sec")
                        msg = f"State: {state}. Resolution {res}x{res} Step {step}: g_loss {self.g_loss:.4f} d_loss {self.d_loss:.4f}"
                        print(msg)

                        self.checkpoint(self.val_z, log2_res, step)
                    self.train_step(data_gen, alpha)

            if log2_res != self.log2_resolution:
                self.grow_model(log2_res+1)
