from keras.models import load_model
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




#내가 구한 것 (not except)
u_valiAEMean = [0.00032736, 0.00042464, 0.00020254]
u_Inv_Cov = np.array([[10894227.3040371 , -1852964.81081921, -2663715.95210474],
                    [-1852964.81081921,  6992086.16355535, -2536576.936831  ],
                    [-2663715.95210474, -2536576.936831  ,  7843481.51137087]])

Unet_model_path = 'F:/SMJ/2023.08.08(Unet)/1D conv_Unet_idx2_(lon_train)/seed_4_LR_8e-05_batch size_16/model_check_point/495-0.00030393.hdf5'
Unet_model = load_model(Unet_model_path)

n = 15
diff = 1
nth_detection_index = 3
threshold = 2


u_MA_save_dir = 'F:/SMJ/2023.10.31(논문 그림 모음)/Unet/학습에 사용된 데이터/'
os.makedirs(u_MA_save_dir, exist_ok=True)

# Case
Case2 = ["L_origin_H300_initV15_cmdV5_initPhi0_cmdPhi30_1_30Hz_data",
         "L_origin_H300_initV15_cmdV5_initPhi0_cmdPhi-30_1_30Hz_data",
         "L_origin_H300_initV15_cmdV10_initPhi0_cmdPhi-30_1_30Hz_data",
         "L_origin_H300_initV15_cmdV10_initPhi0_cmdPhi45_1_30Hz_data"]

Case3 = ["L_origin_H300_initV15_cmdV10_initPhi0_cmdPhi30_1_30Hz_data",
         "L_origin_H300_initV15_cmdV10_initPhi0_cmdPhi-45_1_30Hz_data"]

# 입력데이터
hf = h5py.File('F:/SMJ/2023.08.08(Unet)/hdf_file/1D conv_Unet_idx2_(종_test).hdf5', 'r')
if True:
    for initV in range(15, 24, 3):
        for cmdV in range(5, 15, 5):
            for cmdPhi in range(-45, 60, 15):
                for index in range(1, 2):
                    key_name = "L_origin_H300_initV" + str(initV) + "_cmdV" + str(cmdV) + "_initPhi0_cmdPhi" + str(cmdPhi) + "_" + str(index) + "_30Hz_data"
                    print(key_name)

                    input_data = hf.get(key_name)[:, :, :-1]
                    label = hf.get(key_name)[:, -1, -1]
                    indices = np.where(label == 0)[0]
                    stall_point = indices[0] if indices.size > 0 else None

                    u_result = 'F:/SMJ/2023.10.08(Unet 결과 모음)/mahal_root_MA/u_prediction_data.hdf5'
                    hf_u = h5py.File(u_result, 'r')
                    u_o = hf_u.get(key_name)[:]


                    u_Recon = u_o - input_data
                    u_AA = np.abs(u_Recon - u_valiAEMean)
                    u_mahal = np.zeros(u_AA.shape[0])
                    # 마할라노비스 계산
                    for i in range(u_mahal.shape[0]):
                        u_mahal[i] = np.dot(np.dot(u_AA[i][-1][:], u_Inv_Cov), u_AA[i][-1][:].T)

                    u_mahal = np.sqrt(u_mahal)
                    moving_avg = []
                    for i in range(1,len(u_mahal) + 1):
                        if i < n:
                            subset = np.mean(u_mahal[:i])
                        else:
                            subset = np.mean(u_mahal[i -n :i ])
                        moving_avg.append(subset)
                    moving_avg = np.array(moving_avg)

                    moving_avg = diff_value(moving_avg,diff)
                    u_mahal = diff_value(u_mahal, diff)

                    mahal = moving_avg
                    stall_point = stall_point
                    save_dir = u_MA_save_dir
                    color = 'green'
                    ylabel = f"Delta(t)"


                    for j in range(2):
                        fig8, ax8 = plt.subplots()
                        timestep_index = np.arange(u_mahal.shape[0])
                        ax8.plot(timestep_index, mahal, "-", linewidth=2, c=color)
                        ax8.axvline(stall_point, 1, 0.1, color='orange', linestyle='--') if stall_point is not None else None
                        ax8.set_xlabel("time step", fontsize=14)
                        ax8.set_ylabel(ylabel, fontsize=14)
                        ax8.axhline(threshold, 1, 0.1, color='red', linestyle='-')

                        crossing_indices = np.where(mahal > threshold)[0]
                        if len(crossing_indices) > 0:
                            index,first_crossing_index = find_first_consecutive_run(crossing_indices,nth_detection_index)
                            ax8.plot(first_crossing_index, mahal[first_crossing_index], 'o', color='red')
                            ax8.annotate(f'x={first_crossing_index}',
                                         (first_crossing_index, mahal[first_crossing_index]),
                                         textcoords="offset points", xytext=(-10, -10), ha='center')  # x 값 표기

                            if index != 0:
                                for idx in range(index):
                                    ax8.plot(crossing_indices[idx], mahal[crossing_indices[idx]], 'o', color='blue')

                        # 가장 긴 레이블의 길이를 구합니다.
                        longest_label_len = max(len("Upset:"), len("Detection:"), len("Delay:"))

                        # 숫자와 'timestep'을 포함한 부분의 길이를 구합니다. 이는 모든 줄에서 동일할 것입니다.
                        number_str_len = max(len(f"{stall_point} timestep"), len(f"{first_crossing_index} timestep"),
                                             len(f"{first_crossing_index - stall_point} timestep"))

                        # 전체 문자열의 길이를 계산합니다 (왼쪽 레이블의 길이 + 숫자 부분의 길이).
                        total_length = longest_label_len + number_str_len

                        # 포맷팅을 적용합니다.
                        upset_str = f"{'Upset:':<{longest_label_len}}  {stall_point} timestep".ljust(total_length)
                        detection_str = f"{'Detection:':<{longest_label_len}}  {first_crossing_index} timestep".ljust(
                            total_length)
                        delay_str = f"{'Delay:':<{longest_label_len}}  {first_crossing_index - stall_point} timestep".ljust(
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

                        if j == 0:
                            plt.tight_layout()
                            fig8.savefig(save_dir + f'{initV}_{cmdV}_{cmdPhi}.png', dpi=300)
                        elif j == 1:
                            if np.abs(first_crossing_index - stall_point) > 20 and len(crossing_indices) > 0:
                                ax8.set_xlim(stall_point * 2 - first_crossing_index - 5, first_crossing_index + 5)
                            else:
                                ax8.set_xlim(stall_point - 20, stall_point + 20)
                            ax8.set_ylim(-0.1, 20)
                            plt.tight_layout()
                            fig8.savefig(save_dir + f'x_zoom_{initV}_{cmdV}_{cmdPhi}.png', dpi=300)
                        plt.close()


test_list = ['not_used_condition1_data','not_used_condition2_data','used_condition5_data','PILS_data']
test_list = ['not_used_condition1_data']
Case3 = ['used_condition5_data']

for key_name in test_list:
    hf = h5py.File('F:/SMJ/2023.08.08(Unet)/hdf_file/1D conv_Unet_idx2_Test용.hdf5', 'r')
    if key_name == "PILS_data":
        hf = h5py.File("F:/SMJ/2023.08.08(Unet)/hdf_file/PILS/PILS_data.hdf5",'r')

    u_MA_save_dir = f'F:/SMJ/2023.10.31(논문 그림 모음)/Unet/{key_name}/'
    os.makedirs(u_MA_save_dir, exist_ok=True)
    if key_name == "PILS_data":
        input_data = hf.get(key_name)[:]
        stall_point = 10694
    else:
        input_data = hf.get(key_name)[:, :, :-1]
        label = hf.get(key_name)[:, -1, -1]
        indices = np.where(label == 0)[0]
        stall_point = indices[0] if indices.size > 0 else None
    # 예측파일 들고 오기
    u_result = f'F:/SMJ/2023.10.08(Unet 결과 모음)/mahal_root_MA/{key_name}.hdf5'
    hf_u = h5py.File(u_result, 'r')
    u_o = hf_u.get(key_name)[:]
    u_Recon = u_o - input_data
    u_AA = np.abs(u_Recon - u_valiAEMean)
    u_mahal = np.zeros(u_AA.shape[0])

    for i in range(u_mahal.shape[0]):
        u_mahal[i] = np.dot(np.dot(u_AA[i][-1][:], u_Inv_Cov), u_AA[i][-1][:].T)
    u_mahal = np.sqrt(u_mahal)
    moving_avg = []
    for i in range(1, len(u_mahal) + 1):
        if i < n:
            subset = np.mean(u_mahal[:i])
        else:
            subset = np.mean(u_mahal[i - n:i])
        moving_avg.append(subset)
    moving_avg = np.array(moving_avg)

    moving_avg = diff_value(moving_avg, diff)
    mahal = moving_avg
    color = 'green'
    ylabel = f"Delta(t)"

    save_dir =  u_MA_save_dir


    for j in range(2):
        fig8, ax8 = plt.subplots()
        timestep_index = np.arange(u_mahal.shape[0])
        ax8.plot(timestep_index, mahal, "-", linewidth=2, c=color)
        ax8.axvline(stall_point, 1, 0.1, color='orange',
                    linestyle='--') if stall_point is not None else None
        ax8.set_xlabel("time step", fontsize=14)
        ax8.set_ylabel(ylabel, fontsize=14)
        ax8.axhline(threshold, 1, 0.1, color='red', linestyle='-')

        crossing_indices = np.where(mahal > threshold)[0]
        if key_name == 'PILS_data':
            crossing_indices = crossing_indices[5:]
        if len(crossing_indices) > 0:
            index, first_crossing_index = find_first_consecutive_run(crossing_indices, nth_detection_index)
            ax8.plot(first_crossing_index, mahal[first_crossing_index], 'o', color='red')
            ax8.annotate(f'x={first_crossing_index}',
                         (first_crossing_index, mahal[first_crossing_index]),
                         textcoords="offset points", xytext=(-10, -10), ha='center')  # x 값 표기

            # 가장 긴 레이블의 길이를 구합니다.
            longest_label_len = max(len("Upset:"), len("Detection:"), len("Delay:"))

            # 숫자와 'timestep'을 포함한 부분의 길이를 구합니다. 이는 모든 줄에서 동일할 것입니다.
            number_str_len = max(len(f"{stall_point} timestep"), len(f"{first_crossing_index} timestep"),
                                 len(f"{first_crossing_index - stall_point} timestep"))

            # 전체 문자열의 길이를 계산합니다 (왼쪽 레이블의 길이 + 숫자 부분의 길이).
            total_length = longest_label_len + number_str_len

            # 포맷팅을 적용합니다.
            upset_str = f"{'Upset:':<{longest_label_len}}  {stall_point} timestep".ljust(total_length)
            detection_str = f"{'Detection:':<{longest_label_len}}  {first_crossing_index} timestep".ljust(total_length)
            delay_str = f"{'Delay:':<{longest_label_len}}  {first_crossing_index - stall_point} timestep".ljust(
                total_length)

            # 텍스트를 그래프에 추가합니다. y값을 조절하여 텍스트가 겹치지 않도록 합니다.
            ax8.text(0.03, 0.97, upset_str, transform=ax8.transAxes, fontsize=16, fontweight='bold', va='top',
                     ha='left', color='orange')
            ax8.text(0.03, 0.92, detection_str, transform=ax8.transAxes, fontsize=16, fontweight='bold', va='top',
                     ha='left')
            ax8.text(0.03, 0.87, delay_str, transform=ax8.transAxes, fontsize=16, fontweight='bold', va='top',
                     ha='left', color='red')

            if index != 0:
                for idx in range(index):
                    ax8.plot(crossing_indices[idx], mahal[crossing_indices[idx]], 'o', color='blue')

        if key_name in Case3:
            picture_name = f'Case3_{key_name} distance_plot.png'
        else:
            picture_name = f'{key_name} distance_plot.png'

        if key_name != 'PILS_data':
            if j == 0:
                plt.tight_layout()
                fig8.savefig(save_dir + picture_name, dpi=300)
            elif j == 1:
                if np.abs(first_crossing_index - stall_point) > 20 and len(crossing_indices) > 0:
                    ax8.set_xlim(stall_point * 2 - first_crossing_index - 5, first_crossing_index + 5)
                else:
                    ax8.set_xlim(stall_point - 20, stall_point + 20)
                ax8.set_ylim(-0.1, 20)
                plt.tight_layout()
                fig8.savefig(save_dir + f'x_zoom_{picture_name}', dpi=300)
            plt.close()
        else:
            if j == 0:
                plt.tight_layout()
                fig8.savefig(save_dir + picture_name, dpi=300)
            elif j == 1:
                if np.abs(first_crossing_index - stall_point) > 20 and len(crossing_indices) > 0:
                    ax8.set_xlim(stall_point * 2 - first_crossing_index - 5, first_crossing_index + 5)
                else:
                    ax8.set_xlim(stall_point - 20, stall_point + 20)
                ax8.set_ylim(-0.1, 20)
                plt.tight_layout()
                fig8.savefig(save_dir + f'x_zoom_{picture_name}', dpi=300)
            plt.close()

        plt.close()
print('done')



