"""
Synthetic Evaluation of EBEAE with mFLIM & VNIR Dataset
mFLIM --> multispectral Fluorescence Lifetime Imaging Microscopy
VNIR --> Visible Near Infrared
- DUCD (September/2020)
"""
import logging as log
import matplotlib.pyplot as plt
import numpy as np
from EBEAE import ebeae
from mFLIMsynth import mflimsynth
from VNIRsynth import vnirsynth


def abundance_maps(dataset: str, input_args: dict) -> str:
    try:
        if dataset == 'mFLIM':
            # Plot Ground-Truths and Estimated Abundances
            plt.figure(1, figsize=(7, 5))
            for i in range(1, input_args['n_order'] + 1):
                eval(f"plt.subplot(2{input_args['n_order']}{i})")
                eval(
                    f"plt.imshow(a_origin[{i - 1},:].reshape((input_args['n_samples'],input_args['n_samples'])).T,extent = [0,100,100,0],aspect='auto')")
                plt.title(f"Endmember #{i}", fontweight="bold", fontsize=10)
                eval(f"plt.subplot(2{input_args['n_order']}{i + input_args['n_order']})")
                eval(
                    f"plt.imshow(a_normalized[{i - 1},:].reshape((input_args['n_samples'],input_args['n_samples'])).T,extent = [0,100,100,0], aspect='auto')")
                if i == 2:
                    plt.title("EBEAE Estimation", fontweight="bold", fontsize=10)
            plt.yticks(np.arange(0, 101, 20))
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            # plt.colorbar()
            plt.savefig('Abundance_maps_mFLIM.png')
            return log.info("Abundance_maps_mFLIM.png saved")
        elif dataset == 'VNIR':
            plt.figure(1, figsize=(7, 5))
            for i in range(1, input_args['n_order'] + 1):
                eval(f"plt.subplot(2{input_args['n_order']}{i})")
                eval(
                    f"plt.imshow(a_origin[{i - 1},:].reshape((input_args['n_samples'],input_args['n_samples'])).T,extent = [0,100,100,0],aspect='auto')")
                plt.title(f"Endmember #{i}", fontweight="bold", fontsize=10)
                eval(f"plt.subplot(2{input_args['n_order']}{i + input_args['n_order']})")
                eval(
                    f"plt.imshow(a_normalized[{i - 1},:].reshape((input_args['n_samples'],input_args['n_samples'])).T,extent = [0,100,100,0], aspect='auto')")
                if i == 2:
                    plt.title("EBEAE Estimation", fontweight="bold", fontsize=10)
            plt.xticks(np.arange(0, 101, 20))
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            # plt.colorbar()
            plt.savefig('Abundance_maps_VNIR.png')
            return log.info("Abundance_maps_VNIR.png saved")
        else:
            raise Exception("Incorrect dataset, try again.")
    except Exception as error:
        log.error(f"ERROR [Abundance maps function]: {error}")


def endmember_graph(dataset: str, input_args: dict) -> str:
    try:
        # Plot Ground-Truths and Estimated Endmembers
        if dataset == 'mFLIM':
            plt.figure(2, figsize=(7, 5))
            plt.subplot(211)
            colors = ['royalblue', 'orangered', 'goldenrod', 'purple']
            for i in range(input_args['n_order']):
                if i == 0:
                    plt.plot(p_origin[:, i], color=colors[2], linewidth='2')
                elif i == 1:
                    plt.plot(p_origin[:, i], color=colors[1], linewidth='2')
                elif i == 2:
                    plt.plot(p_origin[:, i], color=colors[3], linewidth='2')
                else:
                    plt.plot(p_origin[:, i], color=colors[0], linewidth='2')
            plt.grid(True)
            plt.subplots_adjust(hspace=0.5)
            plt.axis([0, y_measurements - 1, 0, np.max(p_origin)])
            plt.yticks(np.arange(0, 0.030, 0.005))
            plt.xlabel("Time Sample")
            plt.ylabel("Normalized Intensity")
            plt.title("Ground-truth Endmembers", fontweight="bold", fontsize=10)
            plt.legend(["Endmember #1", "Endmember #2", "Endmember #3", "Endmember #4"])
            plt.subplot(212)
            for col, color in enumerate(colors):
                plt.plot(p_matrix[:, col], color=f'{color}', linewidth='2')
            plt.grid(True)
            plt.axis([0, y_measurements - 1, 0, np.max(p_matrix)])
            plt.rc('font', size=12)  # controls default text sizes
            plt.yticks(np.arange(0, 0.030, 0.005))
            plt.xlabel("Time Sample")
            plt.ylabel("Normalized Intensity")
            plt.title("EBEAE Estimation", fontweight="bold", fontsize=10)
            plt.legend(["Endmember #1", "Endmember #2", "Endmember #3", "Endmember #4"])
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            plt.savefig('Endmembers_graphic_mFLIM.png')
            return log.info("Endmembers_graphic_mFLIM.png saved")
        elif dataset == 'VNIR':
            plt.figure(2, figsize=(7, 5))
            plt.subplot(211)
            plt.plot(bands, p_origin)
            plt.grid(True)
            plt.subplots_adjust(hspace=0.5)
            plt.axis([450, 950, 0, np.max(np.max(p_origin))])
            plt.xticks(np.arange(450, 951, 50))
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Normalized Intensity")
            plt.title("Ground-truth Endmembers", fontweight="bold", fontsize=10)
            plt.legend(["Endmember #1", "Endmember #2", "Endmember #3"])
            plt.subplot(212)
            plt.plot(bands, p_matrix)
            plt.grid(True)
            plt.axis([450, 950, 0, np.max(np.max(p_origin))])
            plt.xticks(np.arange(450, 951, 50))
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Normalized Intensity")
            plt.title("EBEAE Estimation", fontweight="bold", fontsize=10)
            plt.legend(["Endmember #1", "Endmember #2", "Endmember #3"])
            plt.savefig('Endmembers_graphic_VNIR.png')
            return log.info("Endmembers_graphic_VNIR.png saved")
        else:
            raise Exception("Incorrect dataset, try again.")
    except Exception as error:
        log.error(f"ERROR [Endmember graph function]: {error}")


if __name__ == '__main__':
    try:
        # Logs configuration
        log.basicConfig(level=log.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
        # Create synthetic mFLIM database
        choice = input("Running EBEAE algorithm test\nAvailable datasets:\n1)mFLIM\n2)VNIR\nPlease select synthetic"
                       "dataset to start:")
        if choice == '1':
            dataset = 'mFLIM'
            y_matrix_origin, p_origin, a_origin, input_args = mflimsynth(n_order=4, n_samples=100, ts=250e-12, snr=45,
                                                                         psnr=15)
            y_measurements, k_spatial_pos = y_matrix_origin.shape
            mflim_parameters = {
                'initcond': 3,
                'rho': 1,
                'lambda_var': 0.1,
                'epsilon': 1e-3,
                'maxiter': 50,
                'parallel': 0,
                'downsampling': 0.5,
                'normalization': 1,
                'display': 0
            }
        elif choice == '2':
            dataset = 'VNIR'
            y_matrix_origin, p_origin, a_origin, input_args = vnirsynth(n_order=3, n_samples=100, snr=60, psnr=30)
            y_measurements, k_spatial_pos = y_matrix_origin.shape
            bands = np.linspace(450, 950, y_measurements)
            vnir_parameters = {
                'initcond': 1,
                'rho': 0.1,
                'lambda_var': 0.15,
                'epsilon': 1e-3,
                'maxiter': 50,
                'parallel': 0,
                'downsampling': 0.5,
                'normalization': 1,
                'display': 0
            }
        else:
            raise Exception("Incorrect dataset, try again.")
    except Exception as error:
        log.error(f"ERROR [Synthetic database]: {error}")
    try:
        # Execute EBEAE Methodology
        log.info(f"""
        Synthetic {dataset} Dataset
        SNR = {input_args['snr']} dB
        PSNR = {input_args['psnr']} dB
        Number of endmembers = {input_args['n_order']}
        EBEAE Analysis""")
        if dataset == 'mFLIM':
            t_ebeae, results = ebeae(y_matrix=y_matrix_origin, n_order=input_args['n_order'],
                                     parameters=mflim_parameters)
        elif dataset == 'VNIR':
            t_ebeae, results = ebeae(y_matrix=y_matrix_origin, n_order=input_args['n_order'],
                                     parameters=vnir_parameters)
        else:
            raise Exception("Unknown dataset!!!")
        p_matrix, a_scaled, a_normalized, yh_matrix, a_time, p_time = results
    except Exception as error:
        log.error(f"ERROR [EBEAE Methodology]: {error}")
    try:
        # Compute Estimation Errors on Abundances, Endmembers, and Execution Time
        ebeae_endmembers = np.array([])
        ebeae_abundances = np.array([])
        for i in range(input_args['n_order']):
            for j in range(input_args['n_order']):
                ebeae_endmembers = np.append(ebeae_endmembers, np.linalg.norm(p_origin[:, i] - p_matrix[:, j]))
                ebeae_abundances = np.append(ebeae_abundances, np.linalg.norm(a_origin[i, :] - a_normalized[j, :]))
        log.info(f"""
        Performance Metrics
        Execution time = {t_ebeae} s
        Estimation Error in Measurements = {np.linalg.norm(y_matrix_origin - yh_matrix, 'fro') / k_spatial_pos}
        Estimation Error in Endmembers = {np.min(ebeae_endmembers) / (2 * input_args['n_order'])}
        Estimation Error in Abundances = {np.min(ebeae_abundances) / (2 * input_args['n_order'])}
        Execution time for Abundances in ALS procedure = {a_time}
        Execution time for Endmembers in ALS procedure = {p_time}""")
    except Exception as error:
        log.error(f"ERROR [Errors estimation]: {error}")
    try:
        abundance_maps(dataset, input_args)
        endmember_graph(dataset, input_args)
    except Exception as error:
        log.error(f"ERROR [Abundance maps and endmembers generation]: {error}")
