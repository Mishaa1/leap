"""
Functions for handling image datasets.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

class TFRecordDataset:
    """
    Dataset class that loads data from tfrecords files.
    """
    def __init__(self, tfrecord_dir, resolution = None, label_file = None, max_label_size  = 0, repeat = True, buffer_mb = 256, num_threads = 2):
        """
        Initializes TFRecordDataset class

        Args:
            tfrecord_dir (string): directory containing a collection of tfrecords files
            resolution (int, optional): dataset resolution, None = autodetect. Defaults to None
            label_file (string, optional): relative path of the labels file, None = autodetect. Defaults to None
            max_label_size (int, optional): 0 = no labels, 'full' = full labels, <int> = N first label components. Defaults to 0
            repeat (bool, optional): whether to repeat dataset indefinitely. Defaults to True
            buffer_mb (int, optional): read buffer size (megabytes). Defaults to 256
            num_threads (int, optional): number of concurrent threads to use for io. Defaults to 2

        """
        self.tfrecord_dir       = tfrecord_dir
        self.resolution         = None
        self.resolution_log2    = None
        self.shape              = []
        self.dtype              = 'float32'
        self.dynamic_range      = [0, 1]
        self.label_file         = label_file
        self.label_size         = None
        self.label_dtype        = None
        self._np_labels         = None
        self._tf_labels_var     = None
        self._tf_labels_dataset = None
        self._tf_datasets       = dict()
        self._tf_iterator       = None
        self._tf_init_ops       = dict()
        self._tf_minibatch_np   = None
        self._cur_minibatch     = -1
        self._cur_lod           = -1

        # List tfrecords files and inspect their shapes.
        assert os.path.isdir(self.tfrecord_dir)
        tfr_files = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.tfrecords')))
        assert len(tfr_files) >= 1
        tfr_shapes = []
        for tfr_file in tfr_files:
            tfr_opt = tf.io.TFRecordOptions(tf.compat.v1.python_io.TFRecordCompressionType.NONE)
            for record in tf.compat.v1.python_io.tf_record_iterator(tfr_file, tfr_opt):
                tfr_shapes.append(self._parse_tfrecord_np(record).shape)
                break

        # Autodetect label filename.
        if self.label_file is None:
            guess = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.labels')))
            if len(guess):
                self.label_file = guess[0]
        elif not os.path.isfile(self.label_file):
            guess = os.path.join(self.tfrecord_dir, self.label_file)
            if os.path.isfile(guess):
                self.label_file = guess

        # Determine shape and resolution.
        max_shape = max(tfr_shapes, key=lambda shape: np.prod(shape))
        self.resolution = resolution if resolution is not None else max_shape[1]
        self.resolution_log2 = int(np.log2(self.resolution))
        self.shape = [max_shape[0], self.resolution, self.resolution]
        tfr_lods = [self.resolution_log2 - int(np.log2(shape[1])) for shape in tfr_shapes]
        assert all(shape[2] == max_shape[2] for shape in tfr_shapes)
        assert all(shape[0] == shape[1] for shape in tfr_shapes)
        assert all(shape[1] == self.resolution // (2**lod) for shape, lod in zip(tfr_shapes, tfr_lods))
        assert all(lod in tfr_lods for lod in range(self.resolution_log2 - 1))

        # Load labels.
        assert max_label_size == 'full' or max_label_size >= 0
        self._np_labels = np.zeros([1<<20, 0], dtype=np.float32)
        if self.label_file is not None and max_label_size != 0:
            self._np_labels = np.load(self.label_file)
            assert self._np_labels.ndim == 2
        if max_label_size != 'full' and self._np_labels.shape[1] > max_label_size:
            self._np_labels = self._np_labels[:, :max_label_size]
        self.label_size = self._np_labels.shape[1]
        self.label_dtype = self._np_labels.dtype.name

        # Build TF expressions.
        tf_labels_init = tf.zeros(self._np_labels.shape, self._np_labels.dtype)
        self._tf_labels_var = tf.Variable(tf_labels_init, name='labels_var')
        self._tf_labels_var.assign(self._np_labels)
        self._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(self._tf_labels_var)
        for tfr_file, tfr_shape, tfr_lod in zip(tfr_files, tfr_shapes, tfr_lods):
            if tfr_lod < 0:
                continue
            dset = tf.data.TFRecordDataset(tfr_file, compression_type='', buffer_size=buffer_mb<<20)
            dset = dset.map(self._parse_tfrecord_tf, num_parallel_calls=num_threads)
            # dset = tf.data.Dataset.zip((dset, self._tf_labels_dataset))
            bytes_per_item = np.prod(tfr_shape) * np.dtype(self.dtype).itemsize
            if repeat:
                dset = dset.repeat()
            self._tf_datasets[tfr_lod] = dset

    def configure(self, minibatch_size, log2_res):
        """
        Configure current data set iterator, should be called before calling get_minibatch

        Args:
            minibatch_size (int): batch size to use
            log2_res (int): resolution to fetch images a
        """
        res_to_lod = {2:4, 3:3, 4:2, 5:1, 6:0}

        lod = int(np.floor(res_to_lod[log2_res]))
        self._tf_datasets[lod] = self._tf_datasets[lod].batch(minibatch_size, drop_remainder=True)
        self._cur_minibatch = minibatch_size
        self._cur_lod = lod
        self._iterator = iter(self._tf_datasets[lod])

    def get_minibatch(self):
        """
        Gets next batch of training data
        """
        return self._iterator.get_next()

    def _parse_tfrecord_tf(self, record):
        """
        Parses a TFRecord and returns tensor containing data
        """
        features = tf.io.parse_single_example(serialized=record, features={
            'shape': tf.io.FixedLenFeature([3], tf.int64),
            'data': tf.io.FixedLenFeature([], tf.string)})
        data = tf.io.decode_raw(features['data'], tf.float32)
        return tf.reshape(data, features['shape'])

    def _parse_tfrecord_np(self, record):
        """
        Parses a TFRecord and returns numpy array containing data
        """
        ex = tf.train.Example()
        ex.ParseFromString(record)
        shape = ex.features.feature['shape'].int64_list.value
        data = ex.features.feature['data'].bytes_list.value[0]
        return np.fromstring(data, np.float32).reshape(shape)

def load_dataset(data_dir):
    """
    Loads a TFRecord dataset from data dir, should be generated using dataset tool in previous step

    Args:
        data_dir (string): path to directory containing TF Records
    """
    dataset = TFRecordDataset(data_dir)
    return dataset

def load_image(res, image_file):
    """
    Loads an image at a given resolution, assumes rgb tiff images.

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

def plot_images(images, log2_res, fname=''):
    """
    Plot grid of sample images

    Args:
        images (Tensor): Image Data
        log2_res (Int): log2 of image resolution, used to rescale images to fit on grid
        fname (str, optional): If not none, saves a copy of the plot to fname. Defaults to ''.
    """
    scales = {2:0.5,
              3:1,
              4:2,
              5:3,
              6:4,
              7:5,
              8:6,
              9:7,
              10:8}
    scale = scales[log2_res]

    grid_col = min(12, int(12//scale))
    grid_row = images.shape[0]//grid_col
    grid_row = min(2, grid_row)

    figure, axarr = plt.subplots(grid_row, grid_col, figsize=(grid_col*scale, grid_row*scale))

    for row in range(grid_row):
        ax = axarr if grid_row == 1 else axarr[row]
        for col in range(grid_col):
            ax[col].imshow(images[row*grid_col + col])
            ax[col].axis('off')

    if fname:
        print("image name", fname)
        figure.savefig(fname)
