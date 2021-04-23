import os
import numpy as np
import argparse
import warnings

from tensorflow.compat.v1.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img

from adv_compression.compressor import Compressor_ICLR2017
from project_utils import load_img_4d, average_over_dataset


def compress(src_dir, dest_dir, compression_model, checkpoint_dir, recon_prefix='recon_', result_path=None):
    # create folder if needed
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # create compressor object
    if compression_model == 'ICLR2017':
        compressor = Compressor_ICLR2017(checkpoint_dir)
    elif compression_model == 'ICLR2018':
        raise ValueError("ICLR2018 compressor class is not implemented yet")
    else:
        raise ValueError("%s is not a supported model indicator. Currently support ICLR2017 and ICLR2018.")

    # get files in orig_img_dir
    files = os.listdir(src_dir)
    img_files = [f for f in files if f.endswith(".JPEG")]

    # main loop
    compression_results = {}

    for f in img_files:
        try:
            print("========compress image %s ===========" % f)

            # compression original image
            orig_img = load_img_4d(src_dir + f)
            orig_actual_bpp, orig_reconstruct_img, orig_eval_bpp_val, orig_mse_val, orig_psnr_val, orig_msssim_val, num_pixels_val = compressor.compress(orig_img, orig_img)

            save_path = dest_dir + recon_prefix + f[:-4] + 'png'
            save_img(save_path, array_to_img(orig_reconstruct_img[0]))

            # add to results
            compression_results[f] = {'orig_bpp': orig_actual_bpp, 'orig_eval_bpp': orig_eval_bpp_val,
                                      'orig_mse': orig_mse_val,
                                      'orig_psnr': orig_psnr_val, 'orig_ssim': orig_msssim_val}
        except:
            warnings.warn(f"got error when compressing {f}")

    # save results
    if result_path:
        np.save(result_path, compression_results)

    return compression_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default='../DATA/kodak_dataset/',
                        help="dir storing original images.")
    parser.add_argument('--dest_dir', type=str, default='./test/')
    parser.add_argument('--result_path', type=str, default='./compression_results.npy',
                        help='dir to save generated substitute images.')
    parser.add_argument('--compression_model', type=str, default='ICLR2017',
                        help="compression model to use. Now support ICLR2017.")
    parser.add_argument('--checkpoint_dir', type=str, default='../kxw_train/train/', help='dir to load trained model.')
    parser.add_argument('--recon_prefix', type=str, default='recon_')


    args = parser.parse_args()

    # create folder if needed
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)

    return args


if __name__ == '__main__':
    args = parse_args()
    compress(src_dir=args.src_dir,
             dest_dir=args.dest_dir,
             compression_model=args.compression_model,
             checkpoint_dir=args.checkpoint_dir,
             recon_prefix=args.recon_prefix,
             result_path=args.result_path)
