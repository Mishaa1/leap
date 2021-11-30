"""
FELICIA client implementation with proGANs
"""
import logging
import tensorflow as tf
import tensorflow_io as tfio

from progan import dataset, progan

class FeliciaClient():
    def __init__(self, client_num, data_dir, res_dir):
        """
        Initializes a FeliciaClient

        Args:
            client_num (int): client identifier, should be unique across all clients
            data_dir (string): path to dir containing train data
            res_dir (string): path to dir where results should be stored
        """
        self.client_num = client_num
        self.progan = None
        self.data_dir = data_dir
        self.res_dir = res_dir
        self.train_datasets = dataset.load_dataset(data_dir)
        self.logger = logging.getLogger(__name__)

        self.logger.info("Done initializing client %d", client_num)

    def clear_model(self):
        """
        Reset model parameters
        """
        self.progan = None

    def checkpoint(self, log2_res, state, step):
        """
        Save a checkpoint of the model and sample images

        Args:
            log2_res (int): current log2 resolution, used for scaling generated images
            step (int): step number, used in saved file names
        """
        res = 2**log2_res
        msg = f"Resolution {res}x{res} State {state} Step {step}: g_loss {tf.reduce_mean(self.progan.g_loss):.4f} d_loss {self.progan.d_loss:.4f}"
        self.logger.info(msg)
        self.progan.checkpoint(state, log2_res, step)

    def set_training_params(self, batch_sizes, start_log2_res, target_res, learning_rate, beta_1, beta_2, epsilon):
        """
        Sets up training params for progan training
        """
        self.logger.info("Setting client %d params", self.client_num)
        self.batch_sizes = batch_sizes
        self.start_log2_res = start_log2_res
        self.log2_res = start_log2_res
        self.target_res = target_res
        self.progan = progan.ProgressiveGAN(
            results_dir=self.res_dir, resolution=target_res, start_log2_res=start_log2_res,
            checkpoint_suffix=self.client_num, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )

        self.logger.info("Loading dataset for client %d", self.client_num)
        self.train_datasets.configure(batch_sizes[start_log2_res], start_log2_res)

        self.logger.info("Finished setting client %d params", self.client_num)

    def train_step(self, alpha, p_disc, dp_labels, priv_ratio):
        """
        Run a single train step

        Args:
            alpha (tensor): fade-in factor
            p_disc (PrivacyDiscriminator): Privacy Discriminator model used by server
            dp_labels (tensor): real labels for privacy discriminator
            priv_ratio (float): privacy lambda value
        """
        real_images = self.train_datasets.get_minibatch()
        if dp_labels is None:
            self.progan.train_step_felicia(real_images, alpha, p_disc, dp_labels, priv_ratio)
        else:
            loss_g, loss_disc, loss_pdisc = self.progan.train_step_felicia(real_images, alpha, p_disc, dp_labels, priv_ratio)
            self.logger.info("Train step result, client num: %d, generator loss: %.4f, discriminator loss: %.4f, privacy discriminator loss: %.4f", self.client_num, loss_g, loss_disc, loss_pdisc)

    def grow_model(self, target_log2_res):
        """
        Grow generator and discriminator of progan

        Args:
            target_log2_res (int): target resolution on log2 scale
        """
        self.progan.grow_model(target_log2_res)
        self.train_datasets.configure(self.batch_sizes[target_log2_res], target_log2_res)
        self.log2_res = target_log2_res

    def get_data_for_dp(self, num_samples):
        """
        Generate fake images for privacy discriminator training

        Args:
            num_samples (int): number of images to generate

        Returns:
            np_arr: fakes images with shape (num_samples, res, res, channels)
        """
        z = tf.random.normal((num_samples, self.progan.z_dim))
        return self.progan.generate(z)

    def _load_img(self, res, image_file):
        """
        Loads an image at a given resolution, assumes rgb tif images.

        Args:
            res (int): resolution to load image at
            image_file (string): Path to image file

        Returns:
            Tensor: 3D Image
        """
        image = tf.io.read_file(image_file)
        image = tfio.experimental.image.decode_tiff(image) # returns 4 channels
        image = image[:, :, :-1]
        image = tf.image.resize(image, [res, res], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.cast(image, tf.float32)
        image = (image /127.5) - 1
        return image
