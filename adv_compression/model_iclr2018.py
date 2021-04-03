import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
import tensorflow.compat.v1.keras as keras
import numpy as np


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


class AnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(AnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_0")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_1")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_2")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
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
            self.num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_2", inverse=True)),
        tfc.SignalConv2D(
            3, (5, 5), name="layer_3", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=None),
    ]
    super(SynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class HyperAnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(HyperAnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
            padding="same_zeros", use_bias=True,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=False,
            activation=None),
    ]
    super(HyperAnalysisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class HyperSynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(HyperSynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            self.num_filters, (3, 3), name="layer_2", corr=False, strides_up=1,
            padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            activation=None),
    ]
    super(HyperSynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


def load_model(checkpoint_dir):
    # Instantiate model.
    analysis_transform = AnalysisTransform(192)
    synthesis_transform = SynthesisTransform(192)
    hyper_analysis_transform = HyperAnalysisTransform(192)
    hyper_synthesis_transform = HyperSynthesisTransform(192)
    entropy_bottleneck = tfc.EntropyBottleneck()

    # contruct keras model
    model_input = keras.layers.Input(shape=(None, None, 3))
    y = analysis_transform(model_input)
    z = hyper_analysis_transform(abs(y))
    z_tilde, z_likelihoods = entropy_bottleneck(z, training=True)
    sigma = hyper_synthesis_transform(z_tilde)
    scale_table = np.exp(np.linspace(
        np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
    conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table)
    y_tilde, y_likelihoods = conditional_bottleneck(y, training=True)
    x_tilde = synthesis_transform(y_tilde)

    # create a session
    sess = tf.Session()
    # load checkpoint
    latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    # create keras model
    compression_model = keras.models.Model(inputs=model_input, outputs=[y_likelihoods, z_likelihoods, x_tilde])

    return sess, compression_model, model_input


def load_model_with_input(checkpoint_dir, input_variable):
    # get start variables
    start_vars = set(x.name for x in tf.global_variables())

    # Instantiate model.
    analysis_transform = AnalysisTransform(192)
    synthesis_transform = SynthesisTransform(192)
    hyper_analysis_transform = HyperAnalysisTransform(192)
    hyper_synthesis_transform = HyperSynthesisTransform(192)
    entropy_bottleneck = tfc.EntropyBottleneck()

    # contruct keras model
    y = analysis_transform(input_variable)
    z = hyper_analysis_transform(abs(y))
    z_tilde, z_likelihoods = entropy_bottleneck(z, training=True)
    sigma = hyper_synthesis_transform(z_tilde)
    scale_table = np.exp(np.linspace(
        np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
    conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table)
    y_tilde, y_likelihoods = conditional_bottleneck(y, training=True)
    x_tilde = synthesis_transform(y_tilde)

    # create a session
    sess = tf.Session()
    # load checkpoint
    latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
    ## we load all variables only for the compression model
    end_vars = tf.global_variables()
    compression_model_vars = [x for x in end_vars if x.name not in start_vars]
    tf.train.Saver(var_list=compression_model_vars).restore(sess, save_path=latest)

    return sess, input_variable, y_likelihoods, z_likelihoods, x_tilde


def load_test_model_graph(checkpoint_dir):
    '''
    model used in test mode. (entropy_bootleneck(training=False)
    '''
    # inputs
    x = tf.placeholder(tf.float32, [1, None, None, 3])
    orig_x = tf.placeholder(tf.float32, [1, None, None, 3])

    # Instantiate model.
    analysis_transform = AnalysisTransform(192)
    synthesis_transform = SynthesisTransform(192)
    hyper_analysis_transform = HyperAnalysisTransform(192)
    hyper_synthesis_transform = HyperSynthesisTransform(192)
    entropy_bottleneck = tfc.EntropyBottleneck()

    # Transform and compress the image.
    y = analysis_transform(x)
    y_shape = tf.shape(y)
    z = hyper_analysis_transform(abs(y))
    z_hat, z_likelihoods = entropy_bottleneck(z, training=False)
    sigma = hyper_synthesis_transform(z_hat)
    sigma = sigma[:, :y_shape[1], :y_shape[2], :]
    scale_table = np.exp(np.linspace(
        np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
    conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table)
    side_string = entropy_bottleneck.compress(z)
    string = conditional_bottleneck.compress(y)

    # Transform the quantized image back (if requested).
    y_hat, y_likelihoods = conditional_bottleneck(y, training=False)
    x_hat = synthesis_transform(y_hat)

    # eval bpp
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)
    eval_bpp = (tf.reduce_sum(tf.log(y_likelihoods)) +
                tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * num_pixels)

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

    return sess, x, orig_x, [string, side_string], eval_bpp, x_hat, mse, psnr, msssim, num_pixels, y, z