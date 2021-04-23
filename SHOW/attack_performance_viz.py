from matplotlib import pyplot as plt
import numpy as np

from project_utils import average_over_dataset


def draw_mse_increase_against_perturb_mse(result_dir_prefix, epsilons=(1, 2, 4, 8), plot_baseline=True):
    # load adv results
    adv_results_by_epsilons = {}
    for epsilon in epsilons:
        adv_result_path = result_dir_prefix + f'epsilon_{epsilon}/adv_compression_results/compression_results.npy'
        adv_result = np.load(adv_result_path, allow_pickle=True)
        adv_result = adv_result.item()
        adv_results_by_epsilons[epsilon] = adv_result


    # load random perturbation results
    if plot_baseline:
        random_results_by_epsilons = {}
        for epsilon in epsilons:
            random_result_path = result_dir_prefix + f'epsilon_{epsilon}/random_compression_results/compression_results.npy'
            rand_result = np.load(random_result_path, allow_pickle=True)
            rand_result = rand_result.item()
            random_results_by_epsilons[epsilon] = rand_result

    # prepare X and Y
    adv_x = []
    adv_y = []

    for epsilon in epsilons:
        result_dict = adv_results_by_epsilons[epsilon]
        mse_increase = average_over_dataset(result_dict, 'mse_increase')
        perturb_mse = average_over_dataset(result_dict, 'perturb_mse')
        adv_x.append(perturb_mse)
        adv_y.append(mse_increase)

    # prepare X and Y for baseline if apply
    if plot_baseline:
        random_x = []
        random_y = []

        for epsilon in epsilons:
            result_dict = random_results_by_epsilons[epsilon]
            mse_increase = average_over_dataset(result_dict, 'mse_increase')
            perturb_mse = average_over_dataset(result_dict, 'perturb_mse')
            random_x.append(perturb_mse)
            random_y.append(mse_increase)

    # plot
    fig, ax = plt.subplots()
    ax.plot(adv_x, adv_y, label="Adv Attack", marker='o')
    if plot_baseline:
        ax.plot(random_x, random_y, label='Rand Attack', marker='^')
    ax.set_xlabel('perturb mse')
    ax.set_ylabel('reconstruction mse increase')
    ax.legend()

    return fig, ax


def draw_rd_tradeoff(result_dir_prefix, llambdas=(0.001, 0.01, 0.1, 1), epsilons=(1,2,4,8), d_metric='psnr'):
    # load results
    compression_results_by_lambda_epsilon = {}
    for llambda in llambdas:
        compression_results_by_lambda_epsilon[llambda] = {}
        for epsilon in epsilons:
            result_path = result_dir_prefix + f'model_lambda_{llambda}/epsilon_{epsilon}/adv_compression_results/compression_results.npy'
            result = np.load(result_path, allow_pickle=True)
            result = result.item()
            compression_results_by_lambda_epsilon[llambda][epsilon] = result

    # plot
    fig, ax = plt.subplots()

    # draw base line
    bpps = [average_over_dataset(compression_results_by_lambda_epsilon[llambda][epsilons[0]], 'orig_bpp') for llambda in llambdas]
    distortion_item = ''
    if d_metric == 'mse':

    distortions = []







