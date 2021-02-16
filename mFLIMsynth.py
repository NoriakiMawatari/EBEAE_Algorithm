# mFLIM dataset synthetic generator
import logging as log
import numpy as np
import scipy.io
from scipy import linalg


def update_args(kwargs: dict) -> dict:
    args_given = list(kwargs.keys())
    default_args = {'n_order': 2, 'n_samples': 60, 'ts': 0.25e-9, 'snr': 0, 'psnr': 0}
    for arg in args_given:
        if arg in default_args.keys():
            default_args[arg] = kwargs[arg]
    return default_args


# Check type-hint
def mflimsynth(**kwargs: [int, float]) -> [np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    INPUTS:
    n_order --> Order of multi-exponential model. Only n = 2,3,4.
    n_samples --> Number of pixels in x & y axes. Size of square image.
    ts --> Sampling Time.
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
            args = {'n_order': 2, 'n_samples': 60, 'ts': 0.25e-9, 'snr': 40, 'psnr': 20}
        elif num_input_args in [1, 2, 3, 4, 5]:
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
        dataexp1 = scipy.io.loadmat("./files/dataexp1")
        u, y = dataexp1["u"], dataexp1["y"]
        del dataexp1
        u_len = len(u)
        u = u/sum(u)  # vector of laser input
        n = np.arange(u_len).reshape((1, u_len))
        t = n*args['ts']  # vector of time samples
        zeros = np.zeros((1, u_len))
        zeros[0][0] = u[0][0]
        u_toe = linalg.toeplitz(u.T, zeros)
        tau11, tau12, tau13, tau14 = [10, 20, 5, 15]
        tau21, tau22, tau23, tau24 = [25, 10, 15, 5]
        tau31, tau32, tau33, tau34 = [5, 7, 35, 15]

        if args['n_order'] == 2:
            aa1 = 7 * np.exp(-0.001 * (xx - num_samples / 2) ** 2 - 0.001 * (yy - num_samples / 2) ** 2) + 0.5
            aa2 = 2.5 * np.exp(-0.001 * (xx - num_samples) ** 2 - 0.001 * yy ** 2) + 2.5 * np.exp(
                -0.0001 * xx ** 2 - 0.001 * (yy - num_samples) ** 2) + \
                  2.5 * np.exp(-0.001 * xx ** 2 - 0.0001 * (yy - num_samples) ** 2) + 2.5 * \
                  np.exp(-0.0001 * (xx - num_samples) ** 2 - 0.001 * (yy - num_samples) ** 2)
            a1 = np.zeros((num_samples, num_samples))
            a2 = np.zeros((num_samples, num_samples))
            for i in np.arange(num_samples):
                for j in np.arange(num_samples):
                    a1[i, j] = aa1[i, j] / (aa1[i, j] + aa2[i, j])
                    a2[i, j] = aa2[i, j] / (aa1[i, j] + aa2[i, j])

        elif args['n_order'] == 3:
            aa1 = 7 * np.exp(-0.005 * (xx - num_samples / 2) ** 2 - 0.005 * (yy - num_samples / 2) ** 2) + 0.5
            aa2 = 2.5 * np.exp(-0.001 * (xx - num_samples) ** 2 - 0.001 * yy ** 2) + \
                  2.5 * np.exp(-0.0001 * xx ** 2 - 0.001 * (yy - num_samples) ** 2)
            aa3 = 3.5 * np.exp(-0.001 * xx ** 2 - 0.0001 * (yy - num_samples) ** 2) + 2.5 * \
                  np.exp(-0.0001 * (xx - num_samples) ** 2 - 0.001 * (yy - num_samples) ** 2)
            a1 = np.zeros((num_samples, num_samples))
            a2 = np.zeros((num_samples, num_samples))
            a3 = np.zeros((num_samples, num_samples))
            for i in np.arange(num_samples):
                for j in np.arange(num_samples):
                    a1[i, j] = aa1[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j])
                    a2[i, j] = aa2[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j])
                    a3[i, j] = aa3[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j])

        elif args['n_order'] == 4:
            aa1 = 2.5 * np.exp(-0.005 * (xx - num_samples / 2) ** 2 - 0.0005 * (yy - num_samples / 2) ** 2) + 0
            # \ + 2.5 * np.exp(-0.0001 * (xx)**2 - 0.001 * (yy - num_samples)**2)
            aa2 = 2.5 * np.exp(-0.001 * (xx - num_samples) ** 2 - 0.00025 * yy ** 2)
            aa3 = 2.5 * np.exp(-0.001 * xx ** 2 - 0.0002 * (yy - num_samples) ** 2)
            aa4 = 2.5 * np.exp(-0.001 * (xx - 8 * (num_samples / 9)) ** 2 - 0.001 * (yy - 8 * (num_samples / 9)) ** 2)\
                  + 2.5 * np.exp(-0.001 * (xx - (num_samples / 9)) ** 2 - 0.001 * (yy - 8 * (num_samples / 9)) ** 2)
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

        p11 = u_toe @ np.exp(-n.T / tau11) * 0.6
        p21 = u_toe @ np.exp(-n.T / tau21) * 0.2
        p31 = u_toe @ np.exp(-n.T / tau31) * 0.2
        p1 = np.r_[p11, p21, p31]

        if args['n_order'] >= 2:
            p12 = u_toe @ np.exp(-n.T / tau12) * 0.2
            p22 = u_toe @ np.exp(-n.T / tau22) * 0.6
            p32 = u_toe @ np.exp(-n.T / tau32) * 0.2
            p2 = np.r_[p12, p22, p32]

            if args['n_order'] >= 3:
                p13 = u_toe @ np.exp(-n.T / tau13) * 0.15
                p23 = u_toe @ np.exp(-n.T / tau23) * 0.15
                p33 = u_toe @ np.exp(-n.T / tau33) * 0.70
                p3 = np.r_[p13, p23, p33]

                if args['n_order'] >= 4:
                    p14 = u_toe @ np.exp(-n.T / tau14) * 0.0
                    p24 = u_toe @ np.exp(-n.T / tau24) * 0.4
                    p34 = u_toe @ np.exp(-n.T / tau34) * 0.6
                    p4 = np.r_[p14, p24, p34]

        y_y = np.zeros((num_samples, num_samples, 3 * u_len))

        for i in range(num_samples):
            for j in range(num_samples):
                if args['n_order'] == 2:
                    y = a1[i, j] * p1 + a2[i, j] * p2
                elif args['n_order'] == 3:
                    y = a1[i, j] * p1 + a2[i, j] * p2 + a3[i, j] * p3
                elif args['n_order'] == 4:
                    y = a1[i, j] * p1 + a2[i, j] * p2 + a3[i, j] * p3 + a4[i, j] * p4

                if noise_measurement and noise_gaussian:
                    sigma_y = np.sqrt((y.T @ y) / (10 ** (args['snr'] / 10)))
                    yy1 = sigma_y * np.random.normal(size=(3 * u_len, 1))
                else:
                    yy1 = np.zeros((3 * u_len, 1))

                if noise_measurement and noise_shot:
                    sigma_y = np.sqrt(np.max(y) / (10 ** (args['psnr'] / 10)))
                    yy2 = sigma_y * np.random.normal(size=(3 * u_len, 1)) * np.sqrt(abs(y))
                else:
                    yy2 = np.zeros((3 * u_len, 1))

                y_y[i, j, :] = y.T + yy1.T + yy2.T

        if args['n_order'] == 2:
            p_matrix = np.c_[p1, p2]
            a_matrix = np.r_[a1.reshape((1, k), order="F"), a2.reshape((1, k), order="F")]
        elif args['n_order'] == 3:
            p_matrix = np.c_[p1, p2, p3]
            a_matrix = np.r_[a1.reshape((1, k), order="F"), a2.reshape(
                (1, k), order="F"), a3.reshape((1, k), order="F")]
        elif args['n_order'] == 4:
            p_matrix = np.c_[p1, p2, p3, p4]
            a_matrix = np.r_[a1.reshape((1, k), order="F"), a2.reshape(
                (1, k), order="F"), a3.reshape((1, k), order="F"), a4.reshape((1, k), order="F")]
        p_sum = np.sum(p_matrix, 0)
        p_matrix = p_matrix / np.tile(p_sum, [p_matrix.shape[0], 1])
        y_matrix_origin = y_y.reshape((k, 3 * u_len), order="F").T
        return y_matrix_origin, p_matrix, a_matrix, args
    except Exception as error:
        log.error(f"ERROR [mFLIM matrices calculation]: {error}")
