from keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Multiply, Activation, Add,concatenate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib,h5py,os,timeit
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#항우연
a_valiAEMean = [0.00224424, 0.00194687, 0.00241044]
a_Inv_Cov = np.array([[324322.14441948, -41443.96395035, -14830.55169333],
                    [-41443.96395035, 311693.02121456, -82298.09629227],
                    [-14830.55169333, -82298.09629227, 307579.18892826]])


autoencoder_model_path = 'F:/SMJ/2023.08.09(autoencoder)/1D conv_Unet_idx2/seed_4_LR_0.0001_batch size_32/model_check_point/88-0.00202166.hdf5'

n = 15     # MA 범위 설정
diff = 1   # 차분값 범위 설정


autoEncoder_model = load_model(autoencoder_model_path)
hf = h5py.File('F:/SMJ/2023.08.09(autoencoder)/hdf_file/for Threshold(all_data + test data).hdf5', 'r')
result = []

for initV in range(15, 24, 3):
    for cmdV in range(5, 15, 5):
        for cmdPhi in range(-45, 60, 15):
            for index in range(1, 2):
                key_name = "L_origin_H300_initV" + str(initV) + "_cmdV" + str(cmdV) + "_initPhi0_cmdPhi" + str(cmdPhi) + "_" + str(index) + "_30Hz_data"
                print(key_name)

                input_data = hf.get(key_name)[:, :, :-1]
                mahal = []
                mahal_MA = []
                mahal_MA_diff = [0]
                running_time = []

                for idx in range(input_data.shape[0]):
                    start = timeit.default_timer()

                    a_o = autoEncoder_model(np.expand_dims(input_data[idx,:,:], axis=0))

                    a_Recon = a_o - input_data[idx,:,:]
                    a_AA = np.abs(a_Recon - a_valiAEMean)

                    a_mahal_inner = np.dot(np.dot(a_AA[0][-1][:], a_Inv_Cov), a_AA[0][-1][:].T)
                    if a_mahal_inner < 0:
                        print("Warning: Value inside the square root is negative!")

                    # 마할라노비스 계산
                    a_mahal = np.sqrt(np.dot(np.dot(a_AA[0][-1][:], a_Inv_Cov), a_AA[0][-1][:].T))
                    end = timeit.default_timer()
                    mahal.append(a_mahal)

                    running_time.append(end - start)
                plt.plot(running_time)
                plt.savefig(f'F:/SMJ/2023.10.31(논문 그림 모음)/running_time/autoencoder/{initV}_{cmdV},{cmdPhi}.png',)
                plt.close()
                result.append(running_time)


