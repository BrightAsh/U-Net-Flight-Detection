from keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Multiply, Activation, Add,concatenate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib,h5py,os,timeit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def diff_value(data,n):
    diff_array = np.zeros(len(data))
    for i in range(len(data)):
        if i < n:
            diff_array[i] = 0
        else:
            diff_array[i] = data[i] - data[i-n]
    return diff_array
def find_first_consecutive_run(arr, n):
    count = 1
    if count == n:
        return 0,arr[0]

    for i in range(1,len(arr)):
        if arr[i] == arr[i-1] + 1:
            count += 1
            if count == n:
                return i,arr[i]
        else:
            count = 1
    return None  # n번 연속되는 구간을 찾지 못했을 경우 None 반환
#항우연
"""

a_valiAEMean = [0.00224424, 0.00194687, 0.00241044]
a_Inv_Cov = np.array([[324322.14441948, -41443.96395035, -14830.55169333],
                    [-41443.96395035, 311693.02121456, -82298.09629227],
                    [-14830.55169333, -82298.09629227, 307579.18892826]])

"""

#항우연(내가 한것)
a_valiAEMean =  [0.00230415, 0.00211229, 0.00265819]
a_Inv_Cov = np.array([[246914.78779075, -66275.16419005, -25370.19992781],
       [-66275.16419005, 213602.62641208, -77047.16127855],
       [-25370.19992781, -77047.16127855, 238461.54595365]])

detection_like_unet = 0
n =15
diff = 1
root_path = 'F:/SMJ/2023.10.31(논문 그림 모음)/종축/Autoencoder/' if detection_like_unet != 1 else  'F:/SMJ/2023.10.31(논문 그림 모음)/종축/Autoencoder_2/'
autoencoder_model_path = 'F:/SMJ/2023.08.09(autoencoder)/1D conv_Unet_idx2/seed_4_LR_0.0001_batch size_32/model_check_point/88-0.00202166.hdf5'
#autoencoder_model_path = 'F:/SMJ/2023.08.09(autoencoder)/1D conv_Unet_idx2/seed_4_LR_0.0001_batch size_32/model_check_point/model_2_lon.hdf5'
autoencoder_model = load_model(autoencoder_model_path)
#1488.2838  #1990.7228
a_threshold = 1990.7229 if detection_like_unet != 1 else 1
a_save_dir = root_path + '학습에 사용된 데이터/'
os.makedirs(a_save_dir, exist_ok=True)
hfa = h5py.File('F:/SMJ/2023.08.09(autoencoder)/hdf_file/for Threshold(all_data + test data).hdf5', 'r')

for initV in range(15, 24, 3):
    for cmdV in range(5, 15, 5):
        for cmdPhi in range(-45, 60, 15):
            for index in range(1, 2):
                key_name = "L_origin_H300_initV" + str(initV) + "_cmdV" + str(cmdV) + "_initPhi0_cmdPhi" + str(cmdPhi) + "_" + str(index) + "_30Hz_data"
                print(key_name)
                max_key_name = "L_origin_H300_initV15_cmdV10_initPhi0_cmdPhi45_1_30Hz_data"
                a_input_data = hfa.get(key_name)[:, :, :-1]
                a_label = hfa.get(key_name)[:, -1, -1]
                a_indices = np.where(a_label == 0)[0]
                a_stall_point = a_indices[0] if a_indices.size > 0 else None



                # # 모델 구동
                # a_o = autoencoder_model.predict(a_input_data, batch_size=32)
                # with h5py.File(a_save_dir + 'a_prediction_data.hdf5', 'a') as f:
                #     if key_name in f:
                #         del f[key_name]
                #     f.create_dataset(key_name, data=a_o)


                #예측파일 들고 오기
                a_result = a_save_dir + 'a_prediction_data.hdf5'
                hf_a = h5py.File(a_result, 'r')
                a_o = hf_a.get(key_name)[:]



                a_Recon = a_o - a_input_data
                a_AA = np.abs(a_Recon - a_valiAEMean)
                a_mahal = np.zeros(a_AA.shape[0]+1)

                for i in range(a_mahal.shape[0]-1):
                    a_mahal[i+1] = np.dot(np.dot(a_AA[i][-1][:], a_Inv_Cov), a_AA[i][-1][:].T)



                if detection_like_unet == 1:
                    a_mahal = np.sqrt(a_mahal)
                    moving_avg = []
                    for i in range(1,len(a_mahal) + 1):
                        if i < n:
                            subset = np.mean(a_mahal[:i])
                        else:
                            subset = np.mean(a_mahal[i -n :i ])
                        moving_avg.append(subset)
                    moving_avg = np.array(moving_avg)

                    moving_avg = diff_value(moving_avg,diff)
                    a_mahal = moving_avg



                color =  'dodgerblue'

                for j in range(3):
                    fig8, ax8 = plt.subplots()
                    timestep_index = np.arange(a_mahal.shape[0])
                    ax8.plot(timestep_index, a_mahal, "-", linewidth=2, c="green")
                    ax8.axvline(a_stall_point, 1, 0.1, color='orange', linestyle='--') if a_stall_point is not None else None
                    if detection_like_unet != 1 and key_name != max_key_name:
                        ax8.axhline(a_mahal[a_stall_point], 1, 0.1, color='red', linestyle='-')
                    ax8.axhline(a_threshold, 1, 0.1, color= color, linestyle='-')
                    ax8.set_xlabel("time step", fontsize=14)
                    ax8.set_ylabel(f"D(t)", fontsize=14)
                    if a_stall_point is not None:
                        crossing_indices = np.where(a_mahal > a_threshold)[0]
                        if detection_like_unet == 1:
                            crossing_indices = crossing_indices[crossing_indices >= 100]
                        if len(crossing_indices) > 0:
                            if detection_like_unet == 1:
                                index, first_crossing_index = find_first_consecutive_run(crossing_indices, 3)
                            else:
                                first_crossing_index = crossing_indices[0]
                            ax8.plot(first_crossing_index, a_mahal[first_crossing_index], 'o', color=color)
                            ax8.annotate(f'x={first_crossing_index}',
                                         (first_crossing_index, a_mahal[first_crossing_index]),
                                         textcoords="offset points", xytext=(-10, -10), ha='center')  # x 값 표기
                            if index != 0 and detection_like_unet == 1:
                                for idx in range(index):
                                    ax8.plot(crossing_indices[idx], a_mahal[crossing_indices[idx]], 'o', color='blue')


                            # 가장 긴 레이블의 길이를 구합니다.
                            longest_label_len = max(len("Upset:"), len("Detection:"), len("Delay:"))

                            # 숫자와 'timestep'을 포함한 부분의 길이를 구합니다. 이는 모든 줄에서 동일할 것입니다.
                            number_str_len = max(len(f"{a_stall_point} timestep"),
                                                 len(f"{first_crossing_index} timestep"),
                                                 len(f"{first_crossing_index - a_stall_point} timestep"))

                            # 전체 문자열의 길이를 계산합니다 (왼쪽 레이블의 길이 + 숫자 부분의 길이).
                            total_length = longest_label_len + number_str_len

                            # 포맷팅을 적용합니다.
                            upset_str = f"{'Upset:':<{longest_label_len}}  {a_stall_point} timestep".ljust(total_length)
                            detection_str = f"{'Detection:':<{longest_label_len}}  {first_crossing_index} timestep".ljust(
                                total_length)
                            delay_str = f"{'Delay:':<{longest_label_len}}  {first_crossing_index - a_stall_point} timestep".ljust(
                                total_length)

                            # 텍스트를 그래프에 추가합니다. y값을 조절하여 텍스트가 겹치지 않도록 합니다.
                            ax8.text(0.03, 0.97, upset_str, transform=ax8.transAxes, fontsize=16, fontweight='bold',
                                     va='top',
                                     ha='left',color = 'orange')
                            ax8.text(0.03, 0.92, detection_str, transform=ax8.transAxes, fontsize=16, fontweight='bold',
                                     va='top',
                                     ha='left')
                            ax8.text(0.03, 0.87, delay_str, transform=ax8.transAxes, fontsize=16, fontweight='bold',
                                     va='top',
                                     ha='left', color='red')
                        ax8.annotate(f'{a_stall_point}', (a_stall_point, ax8.get_ylim()[1] / 2), textcoords="offset points", xytext=(0, 10),ha='center', va='bottom')

                    if j == 0:
                        plt.tight_layout()
                        fig8.savefig(a_save_dir + f'{initV}_{cmdV}_{cmdPhi} distance_plot.png', dpi=300)
                    elif j == 1 and len(crossing_indices) > 0:
                        ax8.set_ylim(-0.01, a_mahal[first_crossing_index] + a_mahal[first_crossing_index]/10)
                        plt.tight_layout()
                        fig8.savefig(a_save_dir + f'y_zoom_{initV}_{cmdV}_{cmdPhi} distance_plot.png', dpi=300)
                    elif j ==2 and len(crossing_indices) > 0:
                        ax8.set_ylim(-0.01, a_mahal[first_crossing_index] + a_mahal[first_crossing_index]/10)
                        ax8.set_xlim(a_stall_point - 50 ,a_stall_point + 20)
                        plt.tight_layout()
                        fig8.savefig(a_save_dir + f'xzoom_{initV}_{cmdV}_{cmdPhi} distance_plot.png', dpi=300)
                    plt.close()

a_hdf_name =  'F:/SMJ/2023.08.09(autoencoder)/hdf_file/for Threshold(all_data + test data).hdf5'
test_name_list = ['not_used_condition1_','not_used_condition2_','used_condition5_']

for sel in range(0,3):
    test_name = test_name_list[sel]
    print(test_name)

    a_save_dir = root_path + f'{test_name}/'
    os.makedirs(a_save_dir, exist_ok=True)

    a_hf = h5py.File(a_hdf_name, 'r')
    a_input_data = a_hf.get(test_name + 'data')[:,:,:-1]
    a_label = a_hf.get(test_name + 'data')[:, -1, -1]
    a_indices = np.where(a_label == 0)[0]
    a_stall_point = a_indices[0] if a_indices.size > 0 else None

    # 예측파일 들고 오기
    a_result = a_save_dir + 'a_prediction_data.hdf5'
    hf_a = h5py.File(a_result, 'r')
    a_o = hf_a.get(key_name)[:]

    a_Recon = a_o - a_input_data
    a_AA = np.abs(a_Recon - a_valiAEMean)
    a_mahal = np.zeros(a_AA.shape[0])
    for i in range(a_mahal.shape[0]):
        a_mahal[i] = np.dot(np.dot(a_AA[i][-1][:], a_Inv_Cov), a_AA[i][-1][:].T)

    if detection_like_unet == 1:
        a_mahal = np.sqrt(a_mahal)
        moving_avg = []
        for i in range(1, len(a_mahal) + 1):
            if i < n:
                subset = np.mean(a_mahal[:i])
            else:
                subset = np.mean(a_mahal[i - n:i])
            moving_avg.append(subset)
        moving_avg = np.array(moving_avg)

        moving_avg = diff_value(moving_avg, diff)
        a_mahal = moving_avg

    color = 'dodgerblue'

    for j in range(3):
        fig8, ax8 = plt.subplots()
        timestep_index = np.arange(a_mahal.shape[0])
        ax8.plot(timestep_index, a_mahal, "-", linewidth=2, c="green")
        ax8.axvline(a_stall_point, 1, 0.1, color='orange', linestyle='--') if a_stall_point is not None else None
        if detection_like_unet != 1 and key_name != max_key_name:
            ax8.axhline(a_mahal[a_stall_point], 1, 0.1, color='red', linestyle='-')
        ax8.axhline(a_threshold, 1, 0.1, color=color, linestyle='-')
        ax8.set_xlabel("time step", fontsize=14)
        ax8.set_ylabel(f"D(t)", fontsize=14)
        if a_stall_point is not None:
            crossing_indices = np.where(a_mahal > a_threshold)[0]
            if detection_like_unet == 1:
                crossing_indices = crossing_indices[crossing_indices >= 100]
            if len(crossing_indices) > 0:
                if detection_like_unet == 1:
                    index, first_crossing_index = find_first_consecutive_run(crossing_indices, 3)
                else:
                    first_crossing_index = crossing_indices[0]
                ax8.plot(first_crossing_index, a_mahal[first_crossing_index], 'o', color=color)
                ax8.annotate(f'x={first_crossing_index}',
                             (first_crossing_index, a_mahal[first_crossing_index]),
                             textcoords="offset points", xytext=(-10, -10), ha='center')  # x 값 표기
                if index != 0 and detection_like_unet == 1:
                    for idx in range(index):
                        ax8.plot(crossing_indices[idx], a_mahal[crossing_indices[idx]], 'o', color='blue')

                # 가장 긴 레이블의 길이를 구합니다.
                longest_label_len = max(len("Upset:"), len("Detection:"), len("Delay:"))

                # 숫자와 'timestep'을 포함한 부분의 길이를 구합니다. 이는 모든 줄에서 동일할 것입니다.
                number_str_len = max(len(f"{a_stall_point} timestep"),
                                     len(f"{first_crossing_index} timestep"),
                                     len(f"{first_crossing_index - a_stall_point} timestep"))

                # 전체 문자열의 길이를 계산합니다 (왼쪽 레이블의 길이 + 숫자 부분의 길이).
                total_length = longest_label_len + number_str_len

                # 포맷팅을 적용합니다.
                upset_str = f"{'Upset:':<{longest_label_len}}  {a_stall_point} timestep".ljust(total_length)
                detection_str = f"{'Detection:':<{longest_label_len}}  {first_crossing_index} timestep".ljust(
                    total_length)
                delay_str = f"{'Delay:':<{longest_label_len}}  {first_crossing_index - a_stall_point} timestep".ljust(
                    total_length)

                # 텍스트를 그래프에 추가합니다. y값을 조절하여 텍스트가 겹치지 않도록 합니다.
                ax8.text(0.03, 0.97, upset_str, transform=ax8.transAxes, fontsize=16, fontweight='bold',
                         va='top',
                         ha='left', color='orange')
                ax8.text(0.03, 0.92, detection_str, transform=ax8.transAxes, fontsize=16, fontweight='bold',
                         va='top',
                         ha='left')
                ax8.text(0.03, 0.87, delay_str, transform=ax8.transAxes, fontsize=16, fontweight='bold',
                         va='top',
                         ha='left', color='red')
            ax8.annotate(f'{a_stall_point}', (a_stall_point, ax8.get_ylim()[1] / 2), textcoords="offset points",
                         xytext=(0, 10), ha='center', va='bottom')

        if j == 0:
            plt.tight_layout()
            fig8.savefig(a_save_dir + f'{initV}_{cmdV}_{cmdPhi} distance_plot.png', dpi=300)
        elif j == 1 and len(crossing_indices) > 0:
            ax8.set_ylim(-0.01, a_mahal[first_crossing_index] + a_mahal[first_crossing_index] / 10)
            plt.tight_layout()
            fig8.savefig(a_save_dir + f'y_zoom_{initV}_{cmdV}_{cmdPhi} distance_plot.png', dpi=300)
        elif j == 2 and len(crossing_indices) > 0:
            ax8.set_ylim(-0.01, a_mahal[first_crossing_index] + a_mahal[first_crossing_index] / 10)
            ax8.set_xlim(a_stall_point - 50, a_stall_point + 20)
            plt.tight_layout()
            fig8.savefig(a_save_dir + f'xzoom_{initV}_{cmdV}_{cmdPhi} distance_plot.png', dpi=300)
        plt.close()