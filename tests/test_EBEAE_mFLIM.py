"""
Synthetic Evaluation of EBEAE with mFLIM Dataset
mFLIM --> multispectral Fluorescence Lifetime Imaging Microscopy
DUCD
September/2020
"""
import numpy as np
from mFLIMsynth import mflimsynth
from EBEAE import ebeae
import matplotlib.pyplot as plt


if __name__ == '__main__':
    try:
        # Input arguments:
        n = 4  # Number of Simulated End-members only n=2,3,4
        nsamples = 100  # Size of the Squared Image nsamples x nsamples
        Ts = 250e-12  # Sampling time
        SNR = 45  # Level in dB of Gaussian Noise SNR=45,50,55
        PSNR = 15  # Level in dB of Shot Noise PSNR=15,20,25

        # Create synthetic mFLIM database
        Y, Po, Ao, u, t = mflimsynth(n, nsamples, Ts, SNR, PSNR)
        L, K = Y.shape

        # Define parameters of EBEAE
        initcond = 3
        rho = 1
        Lambda = 0.1
        epsilon = 1e-3
        maxiter = 50
        parallel = 0
        downsampling = 0.5
        normalization = 1
        display = 0

        # Execute EBEAE Methodology
        print("###################################")
        print("Synthetic mFLIM Dataset")
        print(f"SNR = {SNR} dB")
        print(f"PSNR = {PSNR} dB")
        print(f"Number of endmembers = {n}")
        print("###################################")
        print("EBEAE Analysis")
        parameters = [initcond, rho, Lambda, epsilon, maxiter, downsampling, parallel, normalization, display]
        t_ebeae, results = ebeae(Y, n, parameters, [], [])
        P, A, An, Yh, A_Time, P_Time = results

        # Compute Estimation Errors on Abundances and End-members, and Execution Time
        Ebeae_p = np.array([])
        Ebeae_a = np.array([])
        for i in range(n):
            for j in range(n):
                Ebeae_p = np.append(Ebeae_p, np.linalg.norm(Po[:, i] - P[:, j]))
                Ebeae_a = np.append(Ebeae_a, np.linalg.norm(Ao[i, :] - An[j, :]))
        print("###################")
        print("Performance Metrics")
        print(f"Execution time = {t_ebeae} s")
        print(f"Estimation Error in Measurements = {np.linalg.norm(Y - Yh, 'fro') / K}")
        print(f"Estimation Error in End-members = {np.min(Ebeae_p) / (2 * n)}")
        print(f"Estimation Error in Abundances = {np.min(Ebeae_a) / (2 * n)}")
        print(f"Execution time for Abundances in ALS procedure = {A_Time}")
        print(f"Execution time for Endmembers in ALS procedure = {P_Time}")

        # Plot Ground-Truths and Estimated Abundances
        plt.figure(1, figsize=(7, 5))
        for i in range(1, n+1):
            eval(f"plt.subplot(2{n}{i})")
            eval(f"plt.imshow(Ao[{i - 1},:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')")
            plt.title(f"Perfil #{i}", fontweight="bold")
            eval(f"plt.subplot(2{n}{i+n})")
            eval(f"plt.imshow(An[{i-1},:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0], aspect='auto')")
            if i == 2:
                plt.title("C) Estimación EBEAE en Python", fontweight="bold")
        plt.yticks(np.arange(0, 101, 20))
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.savefig('Abundance_maps_mFLIM.png')

        # Plot Ground-Truths and Estimated Endmembers
        plt.figure(2, figsize=(7, 5))
        plt.subplot(211)
        plt.plot(Po)
        plt.grid(True)
        plt.subplots_adjust(hspace=0.5)
        plt.axis([0, L-1, 0, np.max(Po)])
        plt.xlabel("Tiempo")
        plt.ylabel("Intensidad Normalizada")
        plt.title("Perfiles de Referencia Reales", fontweight="bold")
        plt.legend(["Perfil #1", "Perfil #2", "Perfil #3", "Perfil #4"])
        plt.subplot(212)
        plt.plot(P)
        plt.grid(True)
        plt.axis([0, L-1, 0, np.max(P)])
        plt.xlabel("Tiempo")
        plt.ylabel("Intensidad Normalizada")
        plt.title("C) Estimación EBEAE en Python", fontweight="bold")
        plt.legend(["Perfil #1", "Perfil #2", "Perfil #3", "Perfil #4"])
        plt.savefig('Endmembers_graphic_mFLIM.png')

    except Exception as err:
        print(f'ERROR DETECTED: {err}')
