import os
import numpy as np
import argparse

from tensorflow.compat.v1.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img

from adv_compression.compressor import Compressor_ICLR2017
from project_utils import load_img_4d, average_over_dataset


def compare_orig_adv_compression(orig_img_dir, adv_img_dir, result_dir, compression_model, checkpoint_dir,
                                  adv_prefix='adv_', silent_mode=False, save_image=True):
    # create folder if needed
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # create compressor object
    if compression_model == 'ICLR2017':
        compressor = Compressor_ICLR2017(checkpoint_dir)
    elif compression_model == 'ICLR2018':
        raise ValueError("ICLR2018 compressor class is not implemented yet")
    else:
        raise ValueError("%s is not a supported model indicator. Currently support ICLR2017 and ICLR2018.")

    # get files in orig_img_dir
    files = os.listdir(orig_img_dir)
    img_files = [f for f in files if f.endswith(".png")]

    # main loop
    compression_results = {}

    for f in img_files:
        print("========compress image %s ===========" % f)
        subs_f = adv_prefix + f

        # compression original image
        orig_img = load_img_4d(orig_img_dir + f)
        orig_actual_bpp, orig_reconstruct_img, orig_eval_bpp_val, orig_mse_val, orig_psnr_val, orig_msssim_val, num_pixels_val = compressor.compress(orig_img, orig_img)

        if save_image:
            save_path = result_dir + 'recon_' + f
            save_img(save_path, array_to_img(orig_reconstruct_img[0]))

        # compress substitute image
        adv_img = load_img_4d(adv_img_dir + subs_f)
        adv_actual_bpp, adv_reconstruct_img, adv_eval_bpp_val, adv_mse_val, adv_psnr_val, adv_msssim_val, num_pixels_val = compressor.compress(adv_img, orig_img)

        if save_image:
            save_path = result_dir + 'recon_' + subs_f
            save_img(save_path, array_to_img(adv_reconstruct_img[0]))

        # compute distortions
        adv_mse = (np.square(orig_img * 255 - adv_img * 255)).mean()
        mse_increase = adv_mse_val - orig_mse_val
        psnr_decrease = orig_psnr_val - adv_psnr_val

        # add to results
        compression_results[f] = {'perturb_mse': adv_mse, 'mse_increase': mse_increase, 'psnr_decrease': psnr_decrease,
                                  'orig_bpp': orig_actual_bpp, 'orig_eval_bpp': orig_eval_bpp_val,
                                  'orig_mse': orig_mse_val,
                                  'orig_psnr': orig_psnr_val, 'orig_ssim': orig_msssim_val, 'adv_bpp': adv_actual_bpp,
                                  'adv_actual_bpp': adv_eval_bpp_val, 'adv_mse': adv_mse_val, 'adv_psnr': adv_psnr_val,
                                  'adv_ssim': adv_msssim_val}

        if not silent_mode:
            print(f'adv mse: {adv_mse}')
            print(f'mse increase: {mse_increase}')
            print(f'psnr decrease: {psnr_decrease}')

    # save results
    print('Result over the entire dataset')
    print(f'avg adv mse: {average_over_dataset(compression_results, "perturb_mse")}')
    print(f'avg mse increase: {average_over_dataset(compression_results, "mse_increase")}')
    print(f'avg psnr decrease: {average_over_dataset(compression_results, "mse_increase")}')
    np.save(result_dir + 'compression_results.npy', compression_results)

    return compression_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_img_dir', type=str, default='../DATA/kodak_dataset/',
                        help="dir storing original images.")
    parser.add_argument('--adv_img_dir', type=str, default='./test/')
    parser.add_argument('--result_dir', type=str, default='./exp/',
                        help='dir to save generated substitute images.')
    parser.add_argument('--compression_model', type=str, default='ICLR2017',
                        help="compression model to use. Now support ICLR2017.")
    parser.add_argument('--checkpoint_dir', type=str, default='../kxw_train/train/', help='dir to load trained model.')
    parser.add_argument('--silent_mode', default=False, action='store_true', help="not printing results out")
    parser.add_argument('--save_img', default=False, action='store_true', help='save reconstructed images')

    args = parser.parse_args()

    # create folder if needed
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    return args


if __name__ == '__main__':
    args = parse_args()
    compare_orig_adv_compression(orig_img_dir=args.orig_img_dir,
                                 adv_img_dir=args.adv_img_dir,
                                 result_dir=args.result_dir,
                                 compression_model=args.compression_model,
                                 checkpoint_dir=args.checkpoint_dir,
                                 silent_mode=args.silent_mode,
                                 save_image=args.save_img)

