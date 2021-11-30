"""
Run ProGAN model with kaggle dataset
"""
import random
from glob import glob
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

import config
import dataset
import progan

def main():
    """
    Main workflow
    """
    print("Loading kaggle dataset...")
    train_datasets = {}
    train_images = glob(config.PATH)
    random.shuffle(train_images)
    train_dataset_list = tf.data.Dataset.from_tensor_slices(train_images)

    print("Resizing dataset...")
    n_workers = tf.data.experimental.AUTOTUNE
    for log2_res in range(2, int(np.log2(config.IMAGE_RESOLUTION))+1):
        res = 2**log2_res
        temp = train_dataset_list.map(partial(dataset.load, res), num_parallel_calls=n_workers)

        temp = temp.shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE[log2_res], drop_remainder=True).repeat()
        train_datasets[log2_res] = temp

    print("Setting up GAN...")
    gan = progan.ProgressiveGAN(resolution=config.IMAGE_RESOLUTION, results_dir=config.RESULTS_DIR)

    print("Training GAN...")
    gan.train(train_datasets, config.STEPS_PER_PHASE, config.TICK_INTERVAL)

    for i in range(2, int(np.log2(config.IMAGE_RESOLUTION))):
        gan.to_rgb[i].save(f'{config.RESULTS_DIR}saved_model/to_rgb_{i}')
        gan.from_rgb[i].save(f'{config.RESULTS_DIR}saved_model/from_rgb_{i}')
        gan.discriminator_blocks[i].save(f'{config.RESULTS_DIR}saved_model/d_{i}')
        gan.generator_blocks[i].save(f'{config.RESULTS_DIR}saved_model/g_{i}')

    plot_model(gan.discriminator, show_shapes=True, to_file=f'{config.RESULTS_DIR}discriminator_plot.png')
    plot_model(gan.generator, show_shapes=True, to_file=f'{config.RESULTS_DIR}generator_plot.png')
    plot_model(gan.model, show_shapes=True, to_file=f'{config.RESULTS_DIR}model_plot.png')

    # Inference
    path = f'{config.RESULTS_DIR}saved_model'
    for i in range(2, int(np.log2(config.IMAGE_RESOLUTION))):
        gan.to_rgb[i] = load_model(f'{path}/to_rgb_{i}', compile=False)
        gan.from_rgb[i] = load_model(f'{path}/from_rgb_{i}', compile=False)
        gan.discriminator_blocks[i] = load_model(f'{path}/d_{i}', compile=False)
        gan.generator_blocks[i] = load_model(f'{path}/g_{i}', compile=False)

    log2_res = 6
    gan.grow_model(log2_res)

    gan.alpha = [[1.]]
    latent = tf.random.normal((12, 512))
    images = gan.generate(latent)
    dataset.plot_images(images, log2_res, fname=f'{config.RESULTS_DIR}sample_inference.png')

if __name__ == "__main__":
    main()
