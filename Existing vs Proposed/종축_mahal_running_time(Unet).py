from keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Multiply, Activation, Add,concatenate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib,h5py,os,timeit
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#내가 구한 것 (not except)
u_valiAEMean = [0.00032736, 0.00042464, 0.00020254]
u_Inv_Cov = np.array([[10894227.3040371 , -1852964.81081921, -2663715.95210474],
                    [-1852964.81081921,  6992086.16355535, -2536576.936831  ],
                    [-2663715.95210474, -2536576.936831  ,  7843481.51137087]])

Unet_model_path = 'F:/SMJ/2023.08.08(Unet)/1D conv_Unet_idx2/seed_4_LR_8e-05_batch size_16/model_check_point/495-0.00030393.hdf5'
Unet_model = load_model(Unet_model_path)



n = 15     # MA 범위 설정
diff = 1   # 차분값 범위 설정

# 입력데이터
hf = h5py.File('F:/SMJ/2023.08.08(Unet)/hdf_file/for Threshold(all_data).hdf5', 'r')

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

                    u_o = Unet_model(np.expand_dims(input_data[idx,:,:], axis=0))

                    u_Recon = u_o - input_data[idx,:,:]
                    u_AA = np.abs(u_Recon - u_valiAEMean)

                    u_mahal_inner = np.dot(np.dot(u_AA[0][-1][:], u_Inv_Cov), u_AA[0][-1][:].T)
                    if u_mahal_inner < 0:
                        print("Warning: Value inside the square root is negative!")

                    # 마할라노비스 계산
                    u_mahal = np.sqrt(np.dot(np.dot(u_AA[0][-1][:], u_Inv_Cov), u_AA[0][-1][:].T))
                    mahal.append(u_mahal)

                    # MA 계산
                    if len(mahal) < n:     mahal_MA.append(np.mean(mahal))
                    else:                  mahal_MA.append(np.mean(mahal[-n:]))
                    #diff 계산
                    if len(mahal_MA) > diff:  mahal_MA_diff.append(mahal_MA[-1] - mahal_MA[-diff-1])
                    end = timeit.default_timer()


                    running_time.append(end - start)
                plt.plot(running_time)
                plt.savefig(f'F:/SMJ/2023.10.31(논문 그림 모음)/running_time/Unet/{initV}_{cmdV},{cmdPhi}.png',)
                plt.close()
                result.append(running_time)



test_list = ['not_used_condition1_data','not_used_condition2_data','PILS_data']

test_result = []
for key_name in test_list:

    if key_name == "PILS_data":
        hf = h5py.File("F:/SMJ/2023.08.08(Unet)/hdf_file/PILS/PILS_data.hdf5", 'r')
        input_data = hf.get(key_name)[:]
    else:
        hf = h5py.File('F:/SMJ/2023.08.08(Unet)/hdf_file/1D conv_Unet_idx2_Test용.hdf5', 'r')
        input_data = hf.get(key_name)[:, :, :-1]

    mahal = []
    mahal_MA = []
    mahal_MA_diff = [0]
    running_time = []

    for idx in range(input_data.shape[0]):
        start = timeit.default_timer()

        u_o = Unet_model(np.expand_dims(input_data[idx,:,:], axis=0))

        u_Recon = u_o - input_data[idx, :, :]
        u_AA = np.abs(u_Recon - u_valiAEMean)

        u_mahal_inner = np.dot(np.dot(u_AA[0][-1][:], u_Inv_Cov), u_AA[0][-1][:].T)
        if u_mahal_inner < 0:
            print("Warning: Value inside the square root is negative!")

        # 마할라노비스 계산
        u_mahal = np.sqrt(np.dot(np.dot(u_AA[0][-1][:], u_Inv_Cov), u_AA[0][-1][:].T))
        mahal.append(u_mahal)

        # MA 계산
        if len(mahal) < n:
            mahal_MA.append(np.mean(mahal))
        else:
            mahal_MA.append(np.mean(mahal[-n:]))
        # diff 계산
        if len(mahal_MA) > diff:  mahal_MA_diff.append(mahal_MA[-1] - mahal_MA[-diff - 1])
        end = timeit.default_timer()

        running_time.append(end - start)
    plt.plot(running_time)
    plt.savefig(f'F:/SMJ/2023.10.31(논문 그림 모음)/running_time/Unet/{key_name}.png', )
    plt.close()
    result.append(running_time)

print('done')
