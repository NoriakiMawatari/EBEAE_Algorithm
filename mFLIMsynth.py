# mFLIM dataset synthetic generator
import scipy.io
import numpy as np
from scipy import linalg


def mflimsynth(*args):
    """
    [Y,P,A,u,t] = mflimsynth(n, pix, ts, snr, psnr)
    INPUTS:
    n --> Order of multi-exponential model
    pix --> numbers of pixels in x & y axes
    ts --> Sampling Time
    snr --> snr of Gaussian noise (dB)
    psnr --> psnr for Shot noise (dB)
    OUTPUTS:
    Y --> matrix of fluorescence decays of size 186 x (pix*pix)
    A --> matrix of abundances of n x (pix*pix)
    P --> matrix of end-members 186 x n
    u --> vector of laser input
    t --> vector of time samples
    July/2020
    Daniel U. Campos-Delgado
    """
    # Synthetic FLIM Dataset
    nargin = len(args)
    if nargin == 0:
        N, pix, ts, snr, psnr = [2, 60, 0.25e-9, 40, 20]
    elif nargin < 2:
        N, pix, ts, snr, psnr = [args[0], 60, 0.25e-9, 0, 0]
    elif nargin < 3:
        N, pix, ts, snr, psnr = [args[0], args[1], 0.25e-9, 0, 0]
    elif nargin < 4:
        N, pix, ts, snr, psnr = [args[0], args[1], args[2], 0, 0]
    elif nargin < 5:
        N, pix, ts, snr, psnr = [args[0], args[1], args[2], args[3], 0]
    else:
        N, pix, ts, snr, psnr = list(args)

    if N > 4:
        N = 4
        print("The maximum number of components is 4!!")

    if snr or psnr:
        noise_measurement = True
        if snr:
            noise_gaussian = True
        else:
            noise_gaussian = False
        if psnr:
            noise_shot = True
        else:
            noise_shot = False
    else:
        noise_measurement = False
        noise_gaussian = False
        noise_shot = False

    nsamp = pix
    x = np.arange(1, pix + 1)
    y = np.arange(1, pix + 1)
    xx, yy = np.meshgrid(x, y)
    k = pix * pix
    dataexp1 = scipy.io.loadmat("./files/dataexp1")
    u = dataexp1["u"]
    y = dataexp1["y"]
    del dataexp1
    u_len = len(u)
    u = u / sum(u)
    n = np.arange(u_len).reshape((1, u_len))
    t = n * ts
    zeros = np.zeros((1, u_len))
    zeros[0][0] = u[0][0]
    u_toe = linalg.toeplitz(u.T, zeros)
    tau11, tau12, tau13, tau14 = [10, 20, 5, 15]
    tau21, tau22, tau23, tau24 = [25, 10, 15, 5]
    tau31, tau32, tau33, tau34 = [5, 7, 35, 15]

    if N == 2:
        aa1 = 7 * np.exp(-0.001 * (xx - nsamp / 2) ** 2 - 0.001 * (yy - nsamp / 2) ** 2) + 0.5
        aa2 = 2.5 * np.exp(-0.001 * (xx - nsamp) ** 2 - 0.001 * yy ** 2) + 2.5 * np.exp(
            -0.0001 * xx ** 2 - 0.001 * (yy - nsamp) ** 2) + \
              2.5 * np.exp(-0.001 * xx ** 2 - 0.0001 * (yy - nsamp) ** 2) + 2.5 * \
              np.exp(-0.0001 * (xx - nsamp) ** 2 - 0.001 * (yy - nsamp) ** 2)
        a1 = np.zeros((nsamp, nsamp))
        a2 = np.zeros((nsamp, nsamp))
        for i in np.arange(nsamp):
            for j in np.arange(nsamp):
                a1[i, j] = aa1[i, j] / (aa1[i, j] + aa2[i, j])
                a2[i, j] = aa2[i, j] / (aa1[i, j] + aa2[i, j])

    elif N == 3:
        aa1 = 7 * np.exp(-0.005 * (xx - nsamp / 2) ** 2 - 0.005 * (yy - nsamp / 2) ** 2) + 0.5
        aa2 = 2.5 * np.exp(-0.001 * (xx - nsamp) ** 2 - 0.001 * yy ** 2) + \
              2.5 * np.exp(-0.0001 * xx ** 2 - 0.001 * (yy - nsamp) ** 2)
        aa3 = 3.5 * np.exp(-0.001 * xx ** 2 - 0.0001 * (yy - nsamp) ** 2) + 2.5 * \
              np.exp(-0.0001 * (xx - nsamp) ** 2 - 0.001 * (yy - nsamp) ** 2)
        a1 = np.zeros((nsamp, nsamp))
        a2 = np.zeros((nsamp, nsamp))
        a3 = np.zeros((nsamp, nsamp))
        for i in np.arange(nsamp):
            for j in np.arange(nsamp):
                a1[i, j] = aa1[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j])
                a2[i, j] = aa2[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j])
                a3[i, j] = aa3[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j])

    elif N == 4:
        aa1 = 2.5 * np.exp(-0.005 * (xx - nsamp / 2) ** 2 - 0.0005 * (yy - nsamp / 2) ** 2) + 0
        # \ + 2.5 * np.exp(-0.0001 * (xx)**2 - 0.001 * (yy - nsamp)**2)
        aa2 = 2.5 * np.exp(-0.001 * (xx - nsamp) ** 2 - 0.00025 * yy ** 2)
        aa3 = 2.5 * np.exp(-0.001 * xx ** 2 - 0.0002 * (yy - nsamp) ** 2)
        aa4 = 2.5 * np.exp(-0.001 * (xx - 8 * (nsamp / 9)) ** 2 - 0.001 * (yy - 8 * (nsamp / 9)) ** 2) + \
              2.5 * np.exp(-0.001 * (xx - (nsamp / 9)) ** 2 - 0.001 * (yy - 8 * (nsamp / 9)) ** 2)
        a1 = np.zeros((nsamp, nsamp))
        a2 = np.zeros((nsamp, nsamp))
        a3 = np.zeros((nsamp, nsamp))
        a4 = np.zeros((nsamp, nsamp))
        for i in np.arange(nsamp):
            for j in np.arange(nsamp):
                a1[i, j] = aa1[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j] + aa4[i, j])
                a2[i, j] = aa2[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j] + aa4[i, j])
                a3[i, j] = aa3[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j] + aa4[i, j])
                a4[i, j] = aa4[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j] + aa4[i, j])

    p11 = u_toe @ np.exp(-n.T / tau11) * 0.6
    p21 = u_toe @ np.exp(-n.T / tau21) * 0.2
    p31 = u_toe @ np.exp(-n.T / tau31) * 0.2
    p1 = np.r_[p11, p21, p31]

    if N >= 2:
        p12 = u_toe @ np.exp(-n.T / tau12) * 0.2
        p22 = u_toe @ np.exp(-n.T / tau22) * 0.6
        p32 = u_toe @ np.exp(-n.T / tau32) * 0.2
        p2 = np.r_[p12, p22, p32]

        if N >= 3:
            p13 = u_toe @ np.exp(-n.T / tau13) * 0.15
            p23 = u_toe @ np.exp(-n.T / tau23) * 0.15
            p33 = u_toe @ np.exp(-n.T / tau33) * 0.70
            p3 = np.r_[p13, p23, p33]

            if N >= 4:
                p14 = u_toe @ np.exp(-n.T / tau14) * 0.0
                p24 = u_toe @ np.exp(-n.T / tau24) * 0.4
                p34 = u_toe @ np.exp(-n.T / tau34) * 0.6
                p4 = np.r_[p14, p24, p34]

    y_y = np.zeros((nsamp, nsamp, 3 * u_len))

    for i in range(nsamp):
        for j in range(nsamp):

            if N == 2:
                y = a1[i, j] * p1 + a2[i, j] * p2
            elif N == 3:
                y = a1[i, j] * p1 + a2[i, j] * p2 + a3[i, j] * p3
            elif N == 4:
                y = a1[i, j] * p1 + a2[i, j] * p2 + a3[i, j] * p3 + a4[i, j] * p4

            if noise_measurement == 1 and noise_gaussian == 1:
                sigmay = np.sqrt((y.T @ y) / (10 ** (snr / 10)))
                yy1 = sigmay * np.random.normal(size=(3 * u_len, 1))
            else:
                yy1 = np.zeros((3 * u_len, 1))

            if noise_measurement == 1 and noise_shot == 1:
                sigmay = np.sqrt(np.max(y) / (10 ** (psnr / 10)))
                yy2 = sigmay * np.random.normal(size=(3 * u_len, 1)) * np.sqrt(abs(y))
            else:
                yy2 = np.zeros((3 * u_len, 1))

            y_y[i, j, :] = y.T + yy1.T + yy2.T

    if N == 2:
        P = np.c_[p1, p2]
        A = np.r_[a1.reshape((1, k), order="F"), a2.reshape((1, k), order="F")]
    elif N == 3:
        P = np.c_[p1, p2, p3]
        A = np.r_[a1.reshape((1, k), order="F"), a2.reshape(
            (1, k), order="F"), a3.reshape((1, k), order="F")]
    elif N == 4:
        P = np.c_[p1, p2, p3, p4]
        A = np.r_[a1.reshape((1, k), order="F"), a2.reshape(
            (1, k), order="F"), a3.reshape((1, k), order="F"), a4.reshape((1, k), order="F")]

    Y = y_y.reshape((k, 3 * u_len), order="F").T

    return Y, P, A, u, t
