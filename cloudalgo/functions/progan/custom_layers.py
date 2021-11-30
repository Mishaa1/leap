"""
Custom layers built on Keras Layer base class for proGAN
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer

class PixelNorm(Layer):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def call(self, input_tensor):
        return input_tensor / tf.math.sqrt(tf.reduce_mean(input_tensor**2, axis=-1, keepdims=True) + self.epsilon)

class MinibatchStd(Layer):
    def __init__(self, group_size=4, epsilon=1e-8):

        super(MinibatchStd, self).__init__()
        self.epsilon = epsilon
        self.group_size = group_size

    def call(self, input_tensor):

        n, h, w, c = input_tensor.shape
        x = tf.reshape(input_tensor, [self.group_size, -1, h, w, c])
        group_mean, group_var = tf.nn.moments(x, axes=(0), keepdims=False)
        group_std = tf.sqrt(group_var + self.epsilon)
        avg_std = tf.reduce_mean(group_std, axis=[1,2,3], keepdims=True)
        x = tf.tile(avg_std, [self.group_size, h, w, 1])

        return tf.concat([input_tensor, x], axis=-1)

class FadeIn(Layer):
    @tf.function
    def call(self, input_alpha, a, b):
        input_alpha = tf.convert_to_tensor(input_alpha)
        alpha = tf.reduce_mean(input_alpha)
        y = alpha * a + (1. - alpha) * b
        return y

class Conv2D(Layer):
    def __init__(self, out_channels, kernel=3, gain=2, **kwargs):
        super(Conv2D, self).__init__(kwargs)
        self.kernel = kernel
        self.out_channels = out_channels
        self.gain = gain
        self.pad = kernel!=1

    def build(self, input_shape):
        self.in_channels = input_shape[-1]

        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.w = self.add_weight(shape=[self.kernel,
                                        self.kernel,
                                        self.in_channels,
                                        self.out_channels],
                                initializer=initializer,
                                trainable=True, name='kernel')

        self.b = self.add_weight(shape=(self.out_channels,),
                                initializer='zeros',
                                trainable=True, name='bias')

        fan_in = self.kernel*self.kernel*self.in_channels
        self.scale = tf.sqrt(self.gain/fan_in)

    def call(self, inputs):
        if self.pad:
            x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        else:
            x = inputs
        output = tf.nn.conv2d(x, self.scale*self.w, strides=1, padding="VALID") + self.b
        return output

class Dense(Layer):
    def __init__(self, units, gain=2, **kwargs):
        super(Dense, self).__init__(kwargs)
        self.units = units
        self.gain = gain

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.w = self.add_weight(shape=[self.in_channels,
                                        self.units],
                                initializer=initializer,
                                trainable=True, name='kernel')

        self.b = self.add_weight(shape=(self.units,),
                                initializer='zeros',
                                trainable=True, name='bias')

        fan_in = self.in_channels
        self.scale = tf.sqrt(self.gain/fan_in)

    #@tf.function
    def call(self, inputs):
        output = tf.matmul(inputs, self.scale*self.w) + self.b
        return output

def wasserstein_loss(y_true, y_pred):
    return -tf.reduce_mean(y_true * y_pred)
