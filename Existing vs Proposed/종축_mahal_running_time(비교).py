from keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Multiply, Activation, Add,concatenate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib,h5py,os,timeit
matplotlib.use('Agg')
import matplotlib.pyplot as plt

###################################           AutoEncoder      ###################################
#항우연
a_valiAEMean = [0.00224424, 0.00194687, 0.00241044]
a_Inv_Cov = np.array([[324322.14441948, -41443.96395035, -14830.55169333],
                    [-41443.96395035, 311693.02121456, -82298.09629227],
                    [-14830.55169333, -82298.09629227, 307579.18892826]])
autoencoder_model_path = 'F:/SMJ/2023.08.09(autoencoder)/1D conv_Unet_idx2/seed_4_LR_0.0001_batch size_32/model_check_point/88-0.00202166.hdf5'
autoEncoder_model = load_model(autoencoder_model_path)
hf = h5py.File('F:/SMJ/2023.08.09(autoencoder)/hdf_file/for Threshold(all_data + test data).hdf5', 'r')
a_threshold = 1990.7229

auto_result = []
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
                    s1_end = timeit.default_timer()

                    a_AA = np.abs(a_Recon - a_valiAEMean)
                    a_mahal= np.dot(np.dot(a_AA[0][-1][:], a_Inv_Cov), a_AA[0][-1][:].T)
                    s2_end = timeit.default_timer()

                    if a_mahal > a_threshold: a=0
                    s3_end = timeit.default_timer()


                    s1 = s1_end - start
                    s2 = s2_end - s1_end
                    s3 = s3_end - s2_end
                    all_time = s3_end - start

                    running_time.append((s1, s2, s3, all_time))
                #plt.plot(running_time)
                #plt.savefig(f'F:/SMJ/2023.10.31(논문 그림 모음)/running_time/autoencoder/{initV}_{cmdV},{cmdPhi}.png',)
                #plt.close()
                auto_result.append(running_time)


###################################           Unet       ###################################

#내가 구한 것 (not except)
u_valiAEMean = [0.00032736, 0.00042464, 0.00020254]
u_Inv_Cov = np.array([[10894227.3040371 , -1852964.81081921, -2663715.95210474],
                    [-1852964.81081921,  6992086.16355535, -2536576.936831  ],
                    [-2663715.95210474, -2536576.936831  ,  7843481.51137087]])

Unet_model_path = 'F:/SMJ/2023.08.08(Unet)/1D conv_Unet_idx2_(lon_train)/seed_4_LR_8e-05_batch size_16/model_check_point/495-0.00030393.hdf5'
Unet_model = load_model(Unet_model_path)



n = 15     # MA 범위 설정
diff = 1   # 차분값 범위 설정

# 입력데이터
hf = h5py.File('F:/SMJ/2023.08.08(Unet)/hdf_file/1D conv_Unet_idx2_(종_test).hdf5', 'r')

unet_result = []

for initV in range(15, 24, 3):
    for cmdV in range(5, 15, 5):
        for cmdPhi in range(-45, 60, 15):
            for index in range(1, 2):
                key_name = "L_origin_H300_initV" + str(initV) + "_cmdV" + str(cmdV) + "_initPhi0_cmdPhi" + str(cmdPhi) + "_" + str(index) + "_30Hz_data"
                print(key_name)

                input_data = hf.get(key_name)[:, :, :-1]
                running_time = []
                mahal_sqrt_stack = np.zeros((15))
                mahal_MA_stack = np.zeros((2))
                stall = 0


                for idx in range(input_data.shape[0]):
                    start = timeit.default_timer()

                    u_o = Unet_model(np.expand_dims(input_data[idx,:,:], axis=0))
                    u_Recon = u_o - input_data[idx,:,:]
                    s1_end = timeit.default_timer()

                    u_AA = np.abs(u_Recon - u_valiAEMean)
                    u_mahal = np.sqrt(np.dot(np.dot(u_AA[0][-1][:], u_Inv_Cov), u_AA[0][-1][:].T))
                    mahal_sqrt_stack[:-1] = mahal_sqrt_stack[1:]
                    mahal_sqrt_stack[-1] = u_mahal
                    s2_end = timeit.default_timer()

                    mahal_MA = np.mean(mahal_sqrt_stack)
                    mahal_MA_stack[0] = mahal_MA_stack[1]
                    mahal_MA_stack[1] = mahal_MA
                    s3_end = timeit.default_timer()

                    delta = mahal_MA_stack[1] - mahal_MA_stack[0]
                    if delta <= 2:stall = 0
                    else:stall += 1
                    if stall >= 3: a=0
                    s4_end = timeit.default_timer()

                    s1 = s1_end - start
                    s2 = s2_end - s1_end
                    s3 = s3_end - s2_end
                    s4 = s3_end - s3_end
                    all_time = s4_end - start

                    running_time.append((s1, s2, s3, s4, all_time))

                #plt.plot(running_time)
                #plt.savefig(f'F:/SMJ/2023.10.31(논문 그림 모음)/running_time/Unet/{initV}_{cmdV},{cmdPhi}.png',)
                #plt.close()
                unet_result.append(running_time)


#################################################################
i = 0
for initV in range(15, 24, 3):
    for cmdV in range(5, 15, 5):
        for cmdPhi in range(-45, 60, 15):
            plt.plot(unet_result[i],color = 'red',label = '제안하는 탐지 기법')
            plt.plot(auto_result[i],color = 'skyblue',label = '기존 탐지 기법')
            plt.legend()
            plt.title('Model Execution Time')
            plt.xlabel('Timestep')
            plt.ylabel('Time')
            plt.tight_layout
            plt.savefig(f'F:/SMJ/2023.10.31(논문 그림 모음)/running_time/비교/{initV}_{cmdV},{cmdPhi}.png',)
            i = i + 1
            plt.close()
#################################################################
#################################################################
#################################################################
#################################################################
#################################################################
with h5py.File('C:/Users/song/Desktop/autoencoder_time.h5', 'w') as hf:
    # 각 시간 측정값을 담을 배열을 초기화합니다.
    s1_values, s2_values, s3_values, all_time_values = [], [], [], []

    # auto_result의 각 요소에서 각 시간 측정값을 추출합니다.
    for running_time in auto_result:
        s1, s2, s3, all_time = zip(*running_time)
        s1_values.extend(s1)
        s2_values.extend(s2)
        s3_values.extend(s3)
        all_time_values.extend(all_time)

    # 각 시간 측정값을 HDF5 파일에 저장합니다.
    hf.create_dataset('s1', data=np.array(s1_values))
    hf.create_dataset('s2', data=np.array(s2_values))
    hf.create_dataset('s3', data=np.array(s3_values))
    hf.create_dataset('all_time', data=np.array(all_time_values))

#################################################################
with h5py.File('C:/Users/song/Desktop/autoencoder_time.h5', 'r') as hf:
    # 각 key에 대해 평균, 최대, 최소값 계산
    for key in ['s1', 's2', 's3', 'all_time']:
        data = np.array(hf[key])  # 데이터셋을 NumPy 배열로 불러옵니다
        mean_val = np.mean(data)  # 평균 계산
        max_val = np.max(data)    # 최대값 계산
        min_val = np.min(data)    # 최소값 계산

        # 결과 출력
        print(f'{key} - 평균: {mean_val}, 최대: {max_val}, 최소: {min_val}')

#################################################################
#################################################################
with h5py.File('C:/Users/song/Desktop/unet_time.h5', 'w') as hf:
    # 각 시간 측정값을 담을 배열을 초기화합니다.
    s1_values, s2_values, s3_values, s4_values, all_time_values = [], [], [], [], []

    # unet_result의 각 요소에서 각 시간 측정값을 추출합니다.
    for running_time in unet_result:
        s1, s2, s3, s4, all_time = zip(*running_time)
        s1_values.extend(s1)
        s2_values.extend(s2)
        s3_values.extend(s3)
        s4_values.extend(s4)
        all_time_values.extend(all_time)

    # 각 시간 측정값을 HDF5 파일에 저장합니다.
    hf.create_dataset('s1', data=np.array(s1_values))
    hf.create_dataset('s2', data=np.array(s2_values))
    hf.create_dataset('s3', data=np.array(s3_values))
    hf.create_dataset('s4', data=np.array(s4_values))
    hf.create_dataset('all_time', data=np.array(all_time_values))

#################################################################
with h5py.File('C:/Users/song/Desktop/unet_time.h5', 'r') as hf:
    # 각 key에 대해 평균, 최대, 최소값 계산
    for key in ['s1', 's2', 's3', 's4', 'all_time']:
        data = np.array(hf[key])  # 데이터셋을 NumPy 배열로 불러옵니다
        mean_val = np.mean(data)  # 평균 계산
        max_val = np.max(data)    # 최대값 계산
        min_val = np.min(data)    # 최소값 계산

        # 결과 출력
        print(f'{key} - 평균: {mean_val}, 최대: {max_val}, 최소: {min_val}')
#################################################################
#################################################################
#################################################################
#################################################################
#################################################################

with open("F:/SMJ/2023.10.31(논문 그림 모음)/running_time/비교/unet_result.txt", "w") as f:
    for item in unet_result:
        f.write("%s\n" % item)

# auto_result 리스트를 "auto_result.txt"에 저장
with open("F:/SMJ/2023.10.31(논문 그림 모음)/running_time/비교/auto_result.txt", "w") as f:
    for item in auto_result:
        f.write("%s\n" % item)

#################################################################

unet_avg = [sum(x) / len(x) for x in unet_result]

# auto_result 리스트의 각 요소의 평균 구하기
auto_avg = [sum(x) / len(x) for x in auto_result]

print("unet_result의 평균:", sum(unet_avg) / len(unet_avg))
print("auto_result의 평균:", sum(auto_avg) / len(auto_avg))

# unet_result 리스트의 각 요소의 최대값 구하기
unet_max = [max(x) for x in unet_result]

# auto_result 리스트의 각 요소의 최대값 구하기
auto_max = [max(x) for x in auto_result]

# unet_result 리스트의 각 요소의 최소값 구하기
unet_min = [min(x) for x in unet_result]

# auto_result 리스트의 각 요소의 최소값 구하기
auto_min = [min(x) for x in auto_result]

# 전체 리스트에 대한 최대값과 최소값 구하기
print("unet_result의 최대값:", max(unet_max))
print("unet_result의 최소값:", min(unet_min))
print("auto_result의 최대값:", max(auto_max))
print("auto_result의 최소값:", min(auto_min))


import ast

# "unet_result.txt"에서 unet_result 리스트로 데이터 불러오기
with open("F:/SMJ/2023.10.31(논문 그림 모음)/running_time/비교/unet_result.txt", "r") as f:
    unet_result = [ast.literal_eval(x.strip()) for x in f if x.strip()]  # 문자열을 실제 리스트로 변환

# "auto_result.txt"에서 auto_result 리스트로 데이터 불러오기
with open("F:/SMJ/2023.10.31(논문 그림 모음)/running_time/비교/auto_result.txt", "r") as f:
    auto_result = [ast.literal_eval(x.strip()) for x in f if x.strip()]  # 문자열을 실제 리스트로 변환


unet_result = np.array(unet_result)
auto_result = np.array(auto_result)