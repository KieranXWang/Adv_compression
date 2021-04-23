import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from adv_compression.model_iclr2017 import load_test_model_graph


class Compressor_ICLR2017:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        sess, x, orig_x, string, eval_bpp, x_hat, mse, psnr, msssim, num_pixels, y = load_test_model_graph(checkpoint_dir)
        self.sess = sess
        self.x = x
        self.orig_x = orig_x
        self.string = string
        self.eval_bpp = eval_bpp
        self.x_hat = x_hat
        self.mse = mse
        self.psnr = psnr
        self.msssim = msssim
        self.num_pixels = num_pixels
        self.y = y
        self.tensors = [string, tf.shape(x)[1:-1], tf.shape(y)[1:-1]]

    def compress(self, input_img, orig_img):
        arrays = self.sess.run(self.tensors, feed_dict={self.x: input_img})
        packed = tfc.PackedTensors()
        packed.pack(self.tensors, arrays)

        eval_bpp_val, mse_val, psnr_val, msssim_val, num_pixels_val, reconstruct_img = self.sess.run(
            [self.eval_bpp, self.mse, self.psnr, self.msssim, self.num_pixels, self.x_hat], feed_dict={self.x: input_img, self.orig_x: orig_img})

        actual_bpp = len(packed.string) * 8 / num_pixels_val

        return actual_bpp, reconstruct_img, eval_bpp_val, mse_val, psnr_val, msssim_val, num_pixels_val

