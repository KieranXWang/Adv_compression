import numpy as np
import os
import argparse

from tensorflow.compat.v1.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img

from adv_compression.adv_generator import AdvGenerator


def generate_advs_same_bpp(orig_img_dir, compression_model, checkpoint_dir, result_dir,
                           epsilon=1 / 255, num_steps=100, step_size=0.0001, use_grad_sign=False,
                           img_size=(512, 768, 3),
                           reconstruction_metric='mse', kappa=1e3, tau=0., bpp_offset=0.):
    # create result_dir if the dir does not exist yet
    if not os.path.exists(result_dir): os.makedirs(result_dir)

    # prepare compression model
    if compression_model == 'ICLR2017':
        from adv_compression.model_iclr2017 import load_model
        sess, keras_compression_model = load_model(checkpoint_dir)
        input_placeholder = None
    elif compression_model == 'ICLR2018':
        from adv_compression.model_iclr2018 import load_model
        sess, keras_compression_model, input_placeholder = load_model(checkpoint_dir)
    else:
        raise ValueError("%s is not a supported compression model" % compression_model)

    # prepare input examples
    files = os.listdir(orig_img_dir)
    img_files = [f for f in files if f.endswith(".png")]

    # create SubstituteGenerator object
    adv_generator = AdvGenerator(keras_compression_model=keras_compression_model,
                                               epsilon=epsilon, num_steps=num_steps, step_size=step_size,
                                               use_grad_sign=use_grad_sign, img_size=img_size,
                                               reconstruct_metric=reconstruction_metric, bpp_target=True, kappa=kappa,
                                               tau=tau, compression_model=compression_model,
                                               input_placeholder=input_placeholder)

    # generate perburbations on images
    orig_images = []
    adv_images = []
    for f in img_files:
        # keras load image
        x_image = load_img(orig_img_dir + f)
        x_image = img_to_array(x_image)
        x_image = x_image / 255
        x_image = np.expand_dims(x_image, 0)

        orig_bpp = adv_generator.get_eval_bpp(sess, x_image)
        print('targeting bpp: %.4f' % orig_bpp)
        adv_image = adv_generator.perturb(sess, orig_image=x_image, bpp_target=orig_bpp+bpp_offset)

        # save generated image
        save_path = result_dir + 'adv_' + f
        save_img(save_path, array_to_img(adv_image[0]))

        orig_images.append(x_image)
        adv_images.append(adv_image)

    return orig_images, adv_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_img_dir', type=str, default='../DATA/kodak_dataset/',
                        help="dir storing original images.")
    parser.add_argument('--img_size', type=int, default=[512, 768, 3], nargs='*',
                        help="image size. images in the given dir must have same sizes.")
    parser.add_argument('--compression_model', type=str, default='ICLR2017',
                        help="compression model to use. Now support ICLR2017.")
    parser.add_argument('--checkpoint_dir', type=str, default='../kxw_train/train/', help='dir to load trained model.')
    parser.add_argument('--result_dir', type=str, default='../EXP/test/',
                        help='dir to save generated substitute images.')
    parser.add_argument('--epsilon', type=float, default=1, help="allowed perturbation per pixel. must in range (0,1).")
    parser.add_argument('--num_steps', type=int, default=100, help="number of steps for generating substitute images.")
    parser.add_argument('--step_size', type=float, default=0.001,
                        help="step size in gradient descent for generating substitute images")
    parser.add_argument('--use_grad_sign', default=False, action='store_true', help="use gradient sign for updating.")
    parser.add_argument('--save_diff', default=False, action='store_true',
                        help="save diff images between original and subs")
    parser.add_argument('--reconstruction_metric', type=str, default='mse',
                        help="metric for reconstruction evaluation, mse or ssim.")
    parser.add_argument('--kappa', default=1e3, type=float,
                        help='karpa term in loss function, the strength to keeping target bpp.')
    parser.add_argument('--tau', default=0., type=float,
                        help='tau term in loss function, the tolerance off set to the target bpp')
    parser.add_argument('--bpp_offset', default=0.0, type=float, help="offset to bpp target")

    args = parser.parse_args()

    # create folder if needed
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    return args


if __name__ == '__main__':
    args = parse_args()
    generate_subs_same_bpp(orig_img_dir=args.orig_img_dir,
                           compression_model=args.compression_model,
                           checkpoint_dir=args.checkpoint_dir,
                           result_dir=args.result_dir,
                           epsilon=args.epsilon,
                           num_steps=args.num_steps,
                           step_size=args.step_size,
                           use_grad_sign=args.use_grad_sign,
                           img_size=args.img_size,
                           reconstruction_metric=args.reconstruction_metric,
                           kappa=args.kappa,
                           tau=args.tau,
                           bpp_offset=args.bpp_offset)