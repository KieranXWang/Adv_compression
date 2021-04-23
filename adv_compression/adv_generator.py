import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
import tensorflow.compat.v1.keras as keras
import numpy as np
import os
import argparse
import time
import warnings

from tensorflow.compat.v1.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img


class AdvGenerator(object):
    def __init__(self, keras_compression_model, llambda=0.01, epsilon=64 / 255, num_steps=100, step_size=0.0001,
                 use_grad_sign=False, img_size=(512, 768, 3), reconstruct_metric='mse', bpp_target=False, kappa=1e3,
                 tau=0., compression_model='ICLR2017', input_placeholder=None):
        self.keras_compression_model = keras_compression_model
        self.llambda = llambda
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.img_size = img_size
        self.bpp_target = bpp_target
        self.kappa = kappa
        self.tau = tau
        self.input_placeholder = input_placeholder
        self.compression_model = compression_model

        # x variable and placeholders
        x_variable = tf.Variable(np.zeros((1, img_size[0], img_size[1], img_size[2])), dtype=tf.float32,
                                 name='modifier')
        x_place = tf.placeholder(tf.float32, [1, None, None, 3])
        x_orig_place = tf.placeholder(tf.float32, [1, None, None, 3])
        n_pixel_place = tf.placeholder(tf.float32, shape=())
        assign_x = tf.assign(x_variable, x_place)
        # x clip ops
        delta = tf.clip_by_value(x_variable, 0, 1) - x_place
        delta = tf.clip_by_value(delta, -epsilon, epsilon)
        do_clip_xs = tf.assign(x_variable, x_place + delta)
        # model output
        if compression_model == 'ICLR2017':
            model_output_likelihood, model_output_x_tilde = keras_compression_model(x_variable)
        elif compression_model == 'ICLR2018':
            y_likelihoods, z_likelihoods, model_output_x_tilde = keras_compression_model(x_variable)
        else:
            raise ValueError("%s is not a supported compression model" % compression_model)

        model_output_x_tilde = model_output_x_tilde[:, :img_size[0], :img_size[1], :]
        # loss
        # d loss
        if reconstruct_metric == 'mse':
            d_loss = tf.reduce_mean(tf.squared_difference(x_orig_place, model_output_x_tilde))
            d_loss *= 255 ** 2  # rescale to 0-255 range instead of 0-1 range
        elif reconstruct_metric == 'ssim':
            ssim = tf.image.ssim_multiscale(x_orig_place, model_output_x_tilde, 1)
            ssim_db = -10 * (tf.log(1 - ssim) / tf.log(10.0))
            d_loss = tf.squeeze(-ssim_db)
        else:
            raise ValueError("metric %s is not supported. Use mse or ssim.")
        # r loss
        if compression_model == 'ICLR2017':
            bpp_loss = tf.reduce_sum(tf.log(model_output_likelihood)) / (-np.log(2) * n_pixel_place)
        elif compression_model == 'ICLR2018':
            bpp_loss = (tf.reduce_sum(tf.log(y_likelihoods)) +
                        tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * n_pixel_place)
        else:
            raise ValueError("%s is not a supported compression model" % compression_model)
        # overall loss
        if bpp_target:
            self.bpp_target = tf.placeholder(tf.float32, shape=())
            ## todo, try to see if it is needed to push bbp_loss to self.bpp_target when bpp_loss is smaller than target
            train_loss = -d_loss + kappa * tf.maximum(bpp_loss - self.bpp_target, tau)
        else:
            ## todo, it loss is not modified yet to increase d_loss
            train_loss = llambda * d_loss + bpp_loss
        # add optimizer and get optimizer-related variables (because we need to re-initialized optimizer for each input)
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(step_size)
        grad, var = optimizer.compute_gradients(train_loss, [x_variable])[0]
        if use_grad_sign:
            train_op = optimizer.apply_gradients([(tf.sign(grad), var)])
        else:
            train_op = optimizer.apply_gradients([(grad, var)])
        end_vars = tf.global_variables()
        optimizer_vars = [x for x in end_vars if x.name not in start_vars]

        # add tensors to self for later use
        self.optimizer_vars = optimizer_vars
        self.x_variable = x_variable
        self.x_place = x_place
        self.x_orig_place = x_orig_place
        self.n_pixel_place = n_pixel_place
        self.assign_x = assign_x
        self.d_loss = d_loss
        self.bpp_loss = bpp_loss
        self.train_loss = train_loss
        self.do_clip_x = do_clip_xs
        self.train_op = train_op

    def perturb(self, sess, orig_image, silent_mode=False, bpp_target=None):
        # assert orig_image shape
        assert len(orig_image.shape) == 4, "orig_image must have 4 dimensions (1, None, None, 3)"

        # get input image dimension
        _, n_rows, n_columns, n_channels = orig_image.shape
        n_pixels = n_rows * n_columns

        # init variables
        sess.run(tf.variables_initializer(self.optimizer_vars))
        sess.run(self.x_variable.initializer)
        sess.run(self.assign_x, {self.x_place: orig_image})

        # prepare feed dict
        train_feed_dict = {self.x_orig_place: orig_image, self.n_pixel_place: n_pixels}
        if self.bpp_target is not False:
            train_feed_dict[self.bpp_target] = bpp_target
        if self.compression_model == 'ICLR2018':
            train_feed_dict[self.input_placeholder] = orig_image

        # loss before training
        loss_val, mse_loss_val, bpp_loss_val = sess.run([self.train_loss, self.d_loss, self.bpp_loss], train_feed_dict)
        if not silent_mode:
            print("loss before training: train_loss=%.6f, d_loss=%.6f, bpp_loss=%.6f" % (
                loss_val, mse_loss_val, bpp_loss_val))

        # main training loop
        for i in range(self.num_steps):
            sess.run(self.train_op, train_feed_dict)
            sess.run(self.do_clip_x, {self.x_place: orig_image})
            if not silent_mode:
                loss_val, mse_loss_val, bpp_loss_val = sess.run([self.train_loss, self.d_loss, self.bpp_loss], train_feed_dict)
                print("loss during training: train_loss=%.6f, d_loss=%.6f, bpp_loss=%.6f" % (loss_val, mse_loss_val, bpp_loss_val))

        # after training
        if not silent_mode:
            if not silent_mode:
                loss_val, mse_loss_val, bpp_loss_val = sess.run([self.train_loss, self.d_loss, self.bpp_loss],
                                                                train_feed_dict)
                print("loss after training: train_loss=%.6f, d_loss=%.6f, bpp_loss=%.6f" % (
                    loss_val, mse_loss_val, bpp_loss_val))

        return sess.run(self.x_variable)

    def get_eval_bpp(self, sess, img):
        # assert orig_image shape
        assert len(img.shape) == 4, "orig_image must have 4 dimensions (1, None, None, 3)"

        # get input image dimension
        _, n_rows, n_columns, n_channels = img.shape
        n_pixels = n_rows * n_columns

        # init variables
        sess.run(tf.variables_initializer(self.optimizer_vars))
        sess.run(self.x_variable.initializer)
        sess.run(self.assign_x, {self.x_place: img})

        # get bpp loss
        feed_dict = {self.x_orig_place: img, self.n_pixel_place: n_pixels}
        if self.compression_model == 'ICLR2018':
            feed_dict[self.input_placeholder] = img

        bpp_loss_val = sess.run(self.bpp_loss, feed_dict)

        return bpp_loss_val

    def perturb_target_bpp(self, sess, orig_image, bpp_target, silent_mode=False):
        warnings.warn('this method is deprecated， use the perturb method when having a target bpp')

        # assert orig_image shape
        assert len(orig_image.shape) == 4, "orig_image must have 4 dimensions (1, None, None, 3)"

        # get input image dimension
        _, n_rows, n_columns, n_channels = orig_image.shape
        n_pixels = n_rows * n_columns

        # init variables
        sess.run(tf.variables_initializer(self.optimizer_vars))
        sess.run(self.x_variable.initializer)
        sess.run(self.assign_x, {self.x_place: orig_image})

        # prepare feed dict
        train_feed_dict = {self.x_orig_place: orig_image, self.n_pixel_place: n_pixels, self.bpp_target: bpp_target}
        if self.compression_model == 'ICLR2018':
            train_feed_dict[self.input_placeholder] = orig_image

        # loss before training
        loss_val, mse_loss_val, bpp_loss_val = sess.run([self.train_loss, self.d_loss, self.bpp_loss], train_feed_dict)
        if not silent_mode:
            print("loss before training: train_loss=%.6f, d_loss=%.6f, bpp_loss=%.6f" % (
                loss_val, mse_loss_val, bpp_loss_val))

        # main training loop
        for i in range(self.num_steps):
            sess.run(self.train_op, train_feed_dict)
            sess.run(self.do_clip_x, {self.x_place: orig_image})
            if not silent_mode:
                c_loss = sess.run(self.train_loss, train_feed_dict)
                print('train loss = %.6f' % c_loss)

        # after training
        if not silent_mode:
            if not silent_mode:
                loss_val, mse_loss_val, bpp_loss_val = sess.run([self.train_loss, self.d_loss, self.bpp_loss],
                                                                train_feed_dict)
                print("loss after training: train_loss=%.6f, d_loss=%.6f, bpp_loss=%.6f" % (
                    loss_val, mse_loss_val, bpp_loss_val))

        return sess.run(self.x_variable)

    def perturb_test_speed(self, sess, orig_image, n_pixels):
        prepare_start_time = time.time()

        # init variables
        sess.run(tf.variables_initializer(self.optimizer_vars))
        sess.run(self.x_variable.initializer)
        sess.run(self.assign_x, {self.x_place: orig_image})

        # prepare feed dict
        train_feed_dict = {self.x_orig_place: orig_image, self.n_pixel_place: n_pixels}

        generate_start_time = time.time()

        # main training loop
        for i in range(self.num_steps):
            sess.run(self.train_op, train_feed_dict)
            sess.run(self.do_clip_x, {self.x_place: orig_image})

        generate_end_time = time.time()
        program_time = generate_end_time - prepare_start_time
        generate_time = generate_end_time - generate_start_time

        return program_time, generate_time


class AdvGeneratorTargetD(object):
    def __init__(self, keras_compression_model, llambda=0.01, epsilon=64/255, num_steps=100, step_size=0.0001,
                 use_grad_sign=False, img_size=(512, 768, 3), reconstruct_metric='mse', compression_model='ICLR2017', input_placeholder=None):
        self.keras_compression_model = keras_compression_model
        self.llambda = llambda
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.img_size = img_size
        self.input_placeholder = input_placeholder
        self.compression_model = compression_model

        # x variable and placeholders
        x_variable = tf.Variable(np.zeros((1, img_size[0], img_size[1], img_size[2])), dtype=tf.float32,
                                 name='modifier')
        x_place = tf.placeholder(tf.float32, [1, None, None, 3])
        x_orig_place = tf.placeholder(tf.float32, [1, None, None, 3])
        n_pixel_place = tf.placeholder(tf.float32, shape=())
        assign_x = tf.assign(x_variable, x_place)
        # x clip ops
        delta = tf.clip_by_value(x_variable, 0, 1) - x_place
        delta = tf.clip_by_value(delta, -epsilon, epsilon)
        do_clip_xs = tf.assign(x_variable, x_place + delta)
        # model output
        if compression_model == 'ICLR2017':
            model_output_likelihood, model_output_x_tilde = keras_compression_model(x_variable)
        elif compression_model == 'ICLR2018':
            y_likelihoods, z_likelihoods, model_output_x_tilde = keras_compression_model(x_variable)
        else:
            raise ValueError("%s is not a supported compression model" % compression_model)

        model_output_x_tilde = model_output_x_tilde[:, :img_size[0], :img_size[1], :]
        # loss
        # d loss
        if reconstruct_metric == 'mse':
            d_loss = tf.reduce_mean(tf.squared_difference(x_orig_place, model_output_x_tilde))
            d_loss *= 255 ** 2  # rescale to 0-255 range instead of 0-1 range
        elif reconstruct_metric == 'ssim':
            ssim = tf.image.ssim_multiscale(x_orig_place, model_output_x_tilde, 1)
            ssim_db = -10 * (tf.log(1 - ssim) / tf.log(10.0))
            d_loss = tf.squeeze(-ssim_db)
        else:
            raise ValueError("metric %s is not supported. Use mse or ssim.")
        # r loss
        if compression_model == 'ICLR2017':
            bpp_loss = tf.reduce_sum(tf.log(model_output_likelihood)) / (-np.log(2) * n_pixel_place)
        elif compression_model == 'ICLR2018':
            bpp_loss = (tf.reduce_sum(tf.log(y_likelihoods)) +
                        tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * n_pixel_place)
        else:
            raise ValueError("%s is not a supported compression model" % compression_model)
        # overall loss
        train_loss = -d_loss
        # add optimizer and get optimizer-related variables (because we need to re-initialized optimizer for each input)
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(step_size)
        grad, var = optimizer.compute_gradients(train_loss, [x_variable])[0]
        if use_grad_sign:
            train_op = optimizer.apply_gradients([(tf.sign(grad), var)])
        else:
            train_op = optimizer.apply_gradients([(grad, var)])
        end_vars = tf.global_variables()
        optimizer_vars = [x for x in end_vars if x.name not in start_vars]

        # add tensors to self for later use
        self.optimizer_vars = optimizer_vars
        self.x_variable = x_variable
        self.x_place = x_place
        self.x_orig_place = x_orig_place
        self.n_pixel_place = n_pixel_place
        self.assign_x = assign_x
        self.d_loss = d_loss
        self.bpp_loss = bpp_loss
        self.train_loss = train_loss
        self.do_clip_x = do_clip_xs
        self.train_op = train_op

    def perturb(self, sess, orig_image, silent_mode=False):
        # assert orig_image shape
        assert len(orig_image.shape) == 4, "orig_image must have 4 dimensions (1, None, None, 3)"

        # get input image dimension
        _, n_rows, n_columns, n_channels = orig_image.shape
        n_pixels = n_rows * n_columns

        # init variables
        sess.run(tf.variables_initializer(self.optimizer_vars))
        sess.run(self.x_variable.initializer)
        sess.run(self.assign_x, {self.x_place: orig_image})

        # prepare feed dict
        train_feed_dict = {self.x_orig_place: orig_image, self.n_pixel_place: n_pixels}
        if self.compression_model == 'ICLR2018':
            train_feed_dict[self.input_placeholder] = orig_image

        # loss before training
        loss_val, mse_loss_val, bpp_loss_val = sess.run([self.train_loss, self.d_loss, self.bpp_loss], train_feed_dict)
        if not silent_mode:
            print("loss before training: train_loss=%.6f, d_loss=%.6f, bpp_loss=%.6f" % (
                loss_val, mse_loss_val, bpp_loss_val))

        # main training loop
        for i in range(self.num_steps):
            sess.run(self.train_op, train_feed_dict)
            sess.run(self.do_clip_x, {self.x_place: orig_image})
            if not silent_mode:
                loss_val, mse_loss_val, bpp_loss_val = sess.run([self.train_loss, self.d_loss, self.bpp_loss], train_feed_dict)
                print("loss during training: train_loss=%.6f, d_loss=%.6f, bpp_loss=%.6f" % (loss_val, mse_loss_val, bpp_loss_val))

        # after training
        if not silent_mode:
            if not silent_mode:
                loss_val, mse_loss_val, bpp_loss_val = sess.run([self.train_loss, self.d_loss, self.bpp_loss],
                                                                train_feed_dict)
                print("loss after training: train_loss=%.6f, d_loss=%.6f, bpp_loss=%.6f" % (
                    loss_val, mse_loss_val, bpp_loss_val))

        return sess.run(self.x_variable)

    def get_eval_bpp(self, sess, img):
        # assert orig_image shape
        assert len(img.shape) == 4, "orig_image must have 4 dimensions (1, None, None, 3)"

        # get input image dimension
        _, n_rows, n_columns, n_channels = img.shape
        n_pixels = n_rows * n_columns

        # init variables
        sess.run(tf.variables_initializer(self.optimizer_vars))
        sess.run(self.x_variable.initializer)
        sess.run(self.assign_x, {self.x_place: img})

        # get bpp loss
        feed_dict = {self.x_orig_place: img, self.n_pixel_place: n_pixels}
        if self.compression_model == 'ICLR2018':
            feed_dict[self.input_placeholder] = img

        bpp_loss_val = sess.run(self.bpp_loss, feed_dict)

        return bpp_loss_val

    def perturb_test_speed(self, sess, orig_image, n_pixels):
        prepare_start_time = time.time()

        # init variables
        sess.run(tf.variables_initializer(self.optimizer_vars))
        sess.run(self.x_variable.initializer)
        sess.run(self.assign_x, {self.x_place: orig_image})

        # prepare feed dict
        train_feed_dict = {self.x_orig_place: orig_image, self.n_pixel_place: n_pixels}

        generate_start_time = time.time()

        # main training loop
        for i in range(self.num_steps):
            sess.run(self.train_op, train_feed_dict)
            sess.run(self.do_clip_x, {self.x_place: orig_image})

        generate_end_time = time.time()
        program_time = generate_end_time - prepare_start_time
        generate_time = generate_end_time - generate_start_time

        return program_time, generate_time