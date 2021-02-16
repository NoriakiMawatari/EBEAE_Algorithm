# VNIR dataset synthetic generator
import logging as log
import numpy as np
import scipy.io


def update_args(kwargs: dict) -> dict:
    args_given = list(kwargs.keys())
    default_args = {'n_order': 2, 'n_samples': 60, 'snr': 0, 'psnr': 0}
    for arg in args_given:
        if arg in default_args.keys():
            default_args[arg] = kwargs[arg]
    return default_args


def vnirsynth(**kwargs: [int, float]) -> [np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    INPUTS:
    n_order --> Order of multi-exponential model. Only n = 2,3,4.
    n_samples --> Number of pixels in x & y axes. Size of square image.
    snr --> Level of Gaussian noise (dB), SNR = 45,50,55.
    psnr --> Level for Shot noise (dB), PSNR = 15,20,25.
    OUTPUTS:
    y_matrix --> matrix of fluorescence decays of size [186 x (n_samples*n_samples)].
    a_matrix --> matrix of abundances of [n x (n_samples*n_samples)].
    p_matrix --> matrix of endmembers [186 x n_order].
    - Daniel U. Campos-Delgado (July/2020).
    """
    try:
        # Synthetic FLIM Dataset
        num_input_args = len(kwargs)
        if num_input_args == 0:
            args = {'n_order': 2, 'n_samples': 60, 'snr': 40, 'psnr': 20}
        elif num_input_args in [1, 2, 3, 4]:
            args = update_args(kwargs)

        if args['n_order'] > 4:
            args['n_order'] = 4
            log.info("The maximum number of components is 4!!")

        if args['snr'] or args['psnr']:
            noise_measurement = True
            if args['snr']:
                noise_gaussian = True
            else:
                noise_gaussian = False
            if args['psnr']:
                noise_shot = True
            else:
                noise_shot = False
        else:
            noise_measurement = noise_gaussian = noise_shot = False
    except Exception as error:
        log.error(f"ERROR [Input arguments step]: {error}")
    try:
        num_samples = args['n_samples']
        x = y = np.arange(1, args['n_samples'] + 1)
        xx, yy = np.meshgrid(x, y)
        k = args['n_samples'] * args['n_samples']
        endmembers_vnir = scipy.io.loadmat("./files/EndMembersVNIR")
        p_matrix = endmembers_vnir["P"]
        del endmembers_vnir
        p_rows = p_matrix.shape[0]
        p_1 = p_matrix[:, 0]
        p_2 = p_matrix[:, 1]
        p_3 = p_matrix[:, 2]
        p_4 = p_matrix[:, 3]

        if args['n_order'] == 2:
            aa1 = 7 * np.exp(-0.001 * (xx - num_samples / 2) ** 2 - 0.001 * (yy - num_samples / 2) ** 2) + 0.5
            aa2 = 2.5 * np.exp(-0.001 * (xx - num_samples) ** 2 - 0.001 * yy ** 2) + 2.5 * np.exp(
                -0.0001 * xx ** 2 - 0.001 * (yy - num_samples) ** 2) + 2.5 * np.exp(
                -0.001 * xx ** 2 - 0.0001 * (yy - num_samples) ** 2) + 2.5 * np.exp(
                -0.0001 * (xx - num_samples) ** 2 - 0.001 * (yy - num_samples) ** 2)
            a1 = np.zeros((num_samples, num_samples))
            a2 = np.zeros((num_samples, num_samples))
            for i in np.arange(num_samples):
                for j in np.arange(num_samples):
                    a1[i, j] = aa1[i, j] / (aa1[i, j] + aa2[i, j])
                    a2[i, j] = aa2[i, j] / (aa1[i, j] + aa2[i, j])

        elif args['n_order'] == 3:
            aa1 = 2.5 * np.exp(-0.0025 * (xx - num_samples / 2) ** 2 - 0.0025 * (yy - num_samples / 2) ** 2) + 0
            aa2 = 2.5 * np.exp(-0.001 * (xx - num_samples) ** 2 - 0.001 * yy ** 2) + 2.5 * np.exp(
                -0.0001 * xx ** 2 - 0.001 * (yy - num_samples) ** 2)
            aa3 = 2.5 * np.exp(-0.001 * xx ** 2 - 0.0001 * (yy - num_samples) ** 2) + 2.5 * np.exp(
                -0.0001 * (xx - num_samples) ** 2 - 0.001 * (yy - num_samples) ** 2)
            a1 = np.zeros((num_samples, num_samples))
            a2 = np.zeros((num_samples, num_samples))
            a3 = np.zeros((num_samples, num_samples))
            for i in np.arange(num_samples):
                for j in np.arange(num_samples):
                    a1[i, j] = aa1[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j])
                    a2[i, j] = aa2[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j])
                    a3[i, j] = aa3[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j])

        elif args['n_order'] == 4:
            aa1 = 2.5 * np.exp(-0.005 * (xx - num_samples / 2) ** 2 - 0.0005 * (yy - num_samples / 2) ** 2) + 0.5
            aa2 = 2.5 * np.exp(-0.001 * (xx - num_samples) ** 2 - 0.0001 * yy ** 2) + 2.5 * np.exp(
                -0.0001 * xx ** 2 - 0.001 * (yy - num_samples) ** 2)
            aa3 = 2.5 * np.exp(-0.001 * xx ** 2 - 0.0001 * (yy - num_samples) ** 2)
            aa4 = 2.5 * np.exp(-0.0001 * (xx - num_samples) ** 2 - 0.001 * (yy - num_samples) ** 2)
            a1 = np.zeros((num_samples, num_samples))
            a2 = np.zeros((num_samples, num_samples))
            a3 = np.zeros((num_samples, num_samples))
            a4 = np.zeros((num_samples, num_samples))
            for i in np.arange(num_samples):
                for j in np.arange(num_samples):
                    a1[i, j] = aa1[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j] + aa4[i, j])
                    a2[i, j] = aa2[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j] + aa4[i, j])
                    a3[i, j] = aa3[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j] + aa4[i, j])
                    a4[i, j] = aa4[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j] + aa4[i, j])

        y_y = np.zeros((num_samples, num_samples, p_rows))

        for i in np.arange(num_samples):
            for j in np.arange(num_samples):
                if args['n_order'] == 2:
                    y = a1[i, j] * p_1 + a2[i, j] * p_2
                elif args['n_order'] == 3:
                    y = a1[i, j] * p_1 + a2[i, j] * p_2 + a3[i, j] * p_3
                elif args['n_order'] == 4:
                    y = a1[i, j] * p_1 + a2[i, j] * p_2 + a3[i, j] * p_3 + a4[i, j] * p_4
                if noise_measurement == 1 and noise_gaussian == 1:
                    sigmay = np.sqrt((y.T @ y) / (10 ** (args['snr'] / 10)))
                    yy1 = sigmay * np.random.normal(size=(p_rows, 1))
                else:
                    yy1 = np.zeros((p_rows, 1))
                if noise_measurement == 1 and noise_shot == 1:
                    sigmay = np.sqrt(np.max(y) / (10 ** (args['psnr'] / 10)))
                    yy2 = sigmay * np.random.normal(size=(p_rows, 1)) * np.sqrt(abs(y)).reshape((p_rows, 1), order="F")
                else:
                    yy2 = np.zeros((p_rows, 1))

                y_y[i, j, :] = y.T + yy1.T + yy2.T

        if args['n_order'] == 2:
            p_matrix_origin = np.c_[p_1, p_2]
            a_matrix = np.r_[a1.reshape((1, k), order="F"), a2.reshape((1, k), order="F")]
        elif args['n_order'] == 3:
            p_matrix_origin = np.c_[p_1, p_2, p_3]
            a_matrix = np.r_[a1.reshape((1, k), order="F"), a2.reshape((1, k), order="F"), a3.reshape((1, k), order="F")]
        elif args['n_order'] == 4:
            p_matrix_origin = np.c_[p_1, p_2, p_3, p_4]
            a_matrix = np.r_[
                a1.reshape((1, k), order="F"), a2.reshape((1, k), order="F"), a3.reshape((1, k), order="F"), a4.reshape(
                    (1, k), order="F")]

        y_matrix = y_y.reshape((k, p_rows), order="F").T
        return y_matrix, p_matrix_origin, a_matrix, args
    except Exception as error:
        log.error(f"ERROR [VNIR matrices calculation]: {error}")

