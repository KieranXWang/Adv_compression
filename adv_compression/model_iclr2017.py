import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
import tensorflow.compat.v1.keras as keras
import numpy as np


class AnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(AnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (9, 9), name="layer_0", corr=True, strides_down=4,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_0")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_1")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=False,
            activation=None),
    ]
    super(AnalysisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor

class SynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(SynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_0", inverse=True)),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_1", inverse=True)),
        tfc.SignalConv2D(
            3, (9, 9), name="layer_2", corr=False, strides_up=4,
            padding="same_zeros", use_bias=True,
            activation=None),
    ]
    super(SynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


def load_model(checkpoint_dir):
    # Instantiate model.
    analysis_transform = AnalysisTransform(128)
    entropy_bottleneck = tfc.EntropyBottleneck()
    synthesis_transform = SynthesisTransform(128)

    # construct keras model
    model_input = keras.layers.Input(shape=(None, None, 3))
    y = analysis_transform(model_input)

    # Transform the quantized image back (if requested).
    y_tilde, likelihoods = entropy_bottleneck(y, training=True)
    x_tilde = synthesis_transform(y_tilde)

    # create a session
    sess = tf.Session()
    # load checkpoint
    latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    # create keras model
    compression_model = keras.models.Model(inputs=model_input, outputs=[likelihoods, x_tilde])

    return sess, compression_model


def load_test_model_graph(checkpoint_dir):
    '''
    model used in test mode. (entropy_bootleneck(training=False)
    '''
    x = tf.placeholder(tf.float32, [1, None, None, 3])
    orig_x = tf.placeholder(tf.float32, [1, None, None, 3])
    x_shape = tf.shape(x)

    # Instantiate model.
    analysis_transform = AnalysisTransform(128)
    entropy_bottleneck = tfc.EntropyBottleneck()
    synthesis_transform = SynthesisTransform(128)

    # Transform and compress the image.
    y = analysis_transform(x)
    string = entropy_bottleneck.compress(y)

    # Transform the quantized image back (if requested).
    y_hat, likelihoods = entropy_bottleneck(y, training=False)
    x_hat = synthesis_transform(y_hat)
    x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]

    # eval bpp
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)
    eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

    # reconstruction metric
    # Bring both images back to 0..255 range.
    orig_x_255 = orig_x * 255
    x_hat = tf.clip_by_value(x_hat, 0, 1)
    x_hat = tf.round(x_hat * 255)
    mse = tf.reduce_mean(tf.squared_difference(orig_x_255, x_hat))
    psnr = tf.squeeze(tf.image.psnr(x_hat, orig_x_255, 255))
    msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, orig_x_255, 255))

    # session
    sess = tf.Session()
    # load graph
    latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)

    return sess, x, orig_x, string, eval_bpp, x_hat, mse, psnr, msssim, num_pixels, y