"""
Synthetic Evaluation of EBEAE with mFLIM Dataset
VNIR --> Visible Near Infrared
DUCD
September/2020
"""
import numpy as np
from VNIRsynth import vnirsynth
from EBEAE import ebeae
import matplotlib.pyplot as plt

if __name__ == '__main__':
    try:
        # Input arguments:
        n = 3  # Number of Simulated End-members only n=2,3,4
        nsamples = 100  # Size of the Squared Image nsamples x nsamples
        SNR = 60  # Level in dB of Gaussian Noise     SNR  = 45,50,55,60
        PSNR = 30  # Level in dB of Poisson/Shot Noise PSNR = 15,20,25,30

        # Create synthetic VNIR database
        Y, Po, Ao = vnirsynth(n, nsamples, SNR, PSNR)  # Synthetic VNIR
        L, K = Y.shape
        bands = np.linspace(450, 950, L)

        # Define parameters of EBEAE
        initcond = 1
        rho = 0.1
        Lambda = 0.15
        epsilon = 1e-3
        maxiter = 50
        parallel = 0
        downsampling = 0.5
        normalization = 1
        display = 0

        # Execute EBEAE Methodology
        print("###################################")
        print("Synthetic VNIR Hyperspectral Dataset")
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
            plt.title(f"Endmember #{i}", fontweight="bold", fontsize=10)
            eval(f"plt.subplot(2{n}{i+n})")
            eval(f"plt.imshow(An[{i-1},:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0], aspect='auto')")
            if i == 2:
                plt.title("EBEAE Estimation", fontweight="bold", fontsize=10)
        plt.xticks(np.arange(0, 101, 20))
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        # plt.colorbar()
        plt.savefig('Abundance_maps_VNIR.png')

        # Plot Ground-Truths and Estimated Endmembers
        plt.figure(2, figsize=(7, 5))
        plt.subplot(211)
        plt.plot(bands, Po)
        plt.grid(True)
        plt.subplots_adjust(hspace=0.5)
        plt.axis([450, 950, 0, np.max(np.max(Po))])
        plt.xticks(np.arange(450, 951, 50))
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalized Intensity")
        plt.title("Ground-truth Endmembers", fontweight="bold", fontsize=10)
        plt.legend(["Endmember #1", "Endmember #2", "Endmember #3"])

        plt.subplot(212)
        plt.plot(bands, P)
        plt.grid(True)
        plt.axis([450, 950, 0, np.max(np.max(Po))])
        plt.xticks(np.arange(450, 951, 50))
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalized Intensity")
        plt.title("EBEAE Estimation", fontweight="bold", fontsize=10)
        plt.legend(["Endmember #1", "Endmember #2", "Endmember #3"])
        plt.savefig('Endmembers_graphic_VNIR.png')

    except Exception as err:
        print(f'ERROR DETECTED: {err}')
