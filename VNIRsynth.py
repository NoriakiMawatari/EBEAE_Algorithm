# VNIR Synthetic Function
import scipy.io
import numpy as np


def vnirsynth(*args):
    """
    Y, Po, A = VNIRsynth(N, pix, snr, psnr)
    INPUTS:
    *args : Type = All of them are integers.
    N --> Order of multi-exponential model
    pix --> numbers of pixels in x & y axes
    snr --> snr of Gaussian noise (dB)
    psnr --> psnr for Shot noise (dB)
    OUTPUTS:
    Y --> matrix of fluorescence decays of size 186 x (píx*pix)
    A --> matrix of abundances of N x (pix*pix)
    Po --> matrix of end-members 186 x N
    July/2020
    DUCD
    """
    # Synthetic VNIR Dataset
    nargin = len(args)
    if nargin == 0:
        N, pix, snr, psnr = [2, 60, 40, 20]
    elif nargin < 2:
        N, pix, snr, psnr = [args[0], 60, 0, 0]
    elif nargin < 3:
        N, pix, snr, psnr = [args[0], args[1], 0, 0]
    elif nargin < 4:
        N, pix, snr, psnr = [args[0], args[1], args[2], 0]
    else:
        N, pix, snr, psnr = list(args)

    if args[0] > 4:
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
    K = pix * pix
    EndMembersVNIR = scipy.io.loadmat("../files/EndMembersVNIR")
    P = EndMembersVNIR["P"]
    del EndMembersVNIR
    L = P.shape[0]
    P1 = P[:, 0]
    P2 = P[:, 1]
    P3 = P[:, 2]
    P4 = P[:, 3]

    if N == 2:
        aa1 = 7 * np.exp(-0.001 * (xx - nsamp / 2) ** 2 - 0.001 * (yy - nsamp / 2) ** 2) + 0.5
        aa2 = 2.5 * np.exp(-0.001 * (xx - nsamp) ** 2 - 0.001 * yy ** 2) + 2.5 * np.exp(
            -0.0001 * xx ** 2 - 0.001 * (yy - nsamp) ** 2) + 2.5 * np.exp(
            -0.001 * xx ** 2 - 0.0001 * (yy - nsamp) ** 2) + 2.5 * np.exp(
            -0.0001 * (xx - nsamp) ** 2 - 0.001 * (yy - nsamp) ** 2)
        a1 = np.zeros((nsamp, nsamp))
        a2 = np.zeros((nsamp, nsamp))
        for i in np.arange(nsamp):
            for j in np.arange(nsamp):
                a1[i, j] = aa1[i, j] / (aa1[i, j] + aa2[i, j])
                a2[i, j] = aa2[i, j] / (aa1[i, j] + aa2[i, j])

    elif N == 3:
        aa1 = 2.5 * np.exp(-0.0025 * (xx - nsamp / 2) ** 2 - 0.0025 * (yy - nsamp / 2) ** 2) + 0
        aa2 = 2.5 * np.exp(-0.001 * (xx - nsamp) ** 2 - 0.001 * yy ** 2) + 2.5 * np.exp(
            -0.0001 * xx ** 2 - 0.001 * (yy - nsamp) ** 2)
        aa3 = 2.5 * np.exp(-0.001 * xx ** 2 - 0.0001 * (yy - nsamp) ** 2) + 2.5 * np.exp(
            -0.0001 * (xx - nsamp) ** 2 - 0.001 * (yy - nsamp) ** 2)
        a1 = np.zeros((nsamp, nsamp))
        a2 = np.zeros((nsamp, nsamp))
        a3 = np.zeros((nsamp, nsamp))
        for i in np.arange(nsamp):
            for j in np.arange(nsamp):
                a1[i, j] = aa1[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j])
                a2[i, j] = aa2[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j])
                a3[i, j] = aa3[i, j] / (aa1[i, j] + aa2[i, j] + aa3[i, j])

    elif N == 4:
        aa1 = 2.5 * np.exp(-0.005 * (xx - nsamp / 2) ** 2 - 0.0005 * (yy - nsamp / 2) ** 2) + 0.5
        aa2 = 2.5 * np.exp(-0.001 * (xx - nsamp) ** 2 - 0.0001 * yy ** 2) + 2.5 * np.exp(
            -0.0001 * xx ** 2 - 0.001 * (yy - nsamp) ** 2)
        aa3 = 2.5 * np.exp(-0.001 * xx ** 2 - 0.0001 * (yy - nsamp) ** 2)
        aa4 = 2.5 * np.exp(-0.0001 * (xx - nsamp) ** 2 - 0.001 * (yy - nsamp) ** 2)
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

    Yy = np.zeros((nsamp, nsamp, L))

    for i in np.arange(nsamp):
        for j in np.arange(nsamp):
            if N == 2:
                y = a1[i, j] * P1 + a2[i, j] * P2
            elif N == 3:
                y = a1[i, j] * P1 + a2[i, j] * P2 + a3[i, j] * P3
            elif N == 4:
                y = a1[i, j] * P1 + a2[i, j] * P2 + a3[i, j] * P3 + a4[i, j] * P4
            if noise_measurement == 1 and noise_gaussian == 1:
                sigmay = np.sqrt((y.T @ y) / (10 ** (snr / 10)))
                yy1 = sigmay * np.random.normal(size=(L, 1))
            else:
                yy1 = np.zeros((L, 1))
            if noise_measurement == 1 and noise_shot == 1:
                sigmay = np.sqrt(np.max(y) / (10 ** (psnr / 10)))
                yy2 = sigmay * np.random.normal(size=(L, 1)) * np.sqrt(abs(y)).reshape((L, 1), order="F")
            else:
                yy2 = np.zeros((L, 1))

            Yy[i, j, :] = y.T + yy1.T + yy2.T

    if N == 2:
        Po = np.c_[P1, P2]
        A = np.r_[a1.reshape((1, K), order="F"), a2.reshape((1, K), order="F")]
    elif N == 3:
        Po = np.c_[P1, P2, P3]
        A = np.r_[a1.reshape((1, K), order="F"), a2.reshape((1, K), order="F"), a3.reshape((1, K), order="F")]
    elif N == 4:
        Po = np.c_[P1, P2, P3, P4]
        A = np.r_[
            a1.reshape((1, K), order="F"), a2.reshape((1, K), order="F"), a3.reshape((1, K), order="F"), a4.reshape(
                (1, K), order="F")]

    Y = Yy.reshape((K, L), order="F").T

    return Y, Po, A
