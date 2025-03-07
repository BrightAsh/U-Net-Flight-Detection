from keras.models import load_model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import time
import re
import ast
import scipy.io
import matplotlib,h5py,os,timeit
matplotlib.use('Agg')

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

def Create_PLIS_Data(root_path):
    alpha = []
    q =[]
    theta = []
    feature = ['alpha','q','theta']
    i = 0
    with open(root_path + "/data/PILS data(7층).txt", "r") as f:
        for line in f:
            i +=1
            all_data = line.strip().split(" ")
            alpha.append(all_data[20])  # alpha
            q.append(all_data[8])  # q
            theta.append(all_data[11])  # theta
            if all_data[3] == '408.48':
                stall_point = i -1
    alpha = np.array(alpha)
    q = np.array(q)
    theta = np.array(theta)
    # origin_data 생성
    origin_data = np.stack((alpha, q, theta), axis=1)
    origin_data = np.array(origin_data.astype(float))


    return origin_data, stall_point

def Create_VA_Data(path,sel):
    if sel == 0:
        root_path = path + "/data/T-2_long_150Hz.mat";
    else:
        root_path = path + "/data/L_origin_H300_initV20_cmdV11_initPhi0_cmdPhi32_150Hz_Cut.mat";

    mat_data = scipy.io.loadmat(root_path)
    flight_data = mat_data['origin_data']
    lon_data = flight_data[:,[2,5,17,49]]
    origin_data = lon_data[0:-1:5,:]
    indices = np.where(origin_data[:,-1] == 0)[0]
    stall_point = indices[0] if indices.size > 0 else None

    return origin_data, stall_point


sel = 0

test_type = ['not_con_1','not_con_2','PILS']

a_root_path = f'C:/Users/song/Desktop/JetsonBoard_result/JetsonBoard_AutoEncoder(lon)/result/{test_type[sel]}/'
u_root_path = f'C:/Users/song/Desktop/JetsonBoard_result/JetsonBoard_UNet(lon)/result/{test_type[sel]}/'

a_all_running_time_path = f'{a_root_path}/all_running_time.txt'
a_result_path = f'{a_root_path}/result.txt'
a_model_running_time_path = f'{a_root_path}/unet_running_time.txt'

u_all_running_time_path = f'{u_root_path}/all_running_time.txt'
u_result_path = f'{u_root_path}/result.txt'
u_model_running_time_path = f'{u_root_path}/unet_running_time.txt'

a_all_running_time = np.loadtxt(a_all_running_time_path)
a_model_running_time = np.loadtxt(a_model_running_time_path)
u_all_running_time = np.loadtxt(u_all_running_time_path)
u_model_running_time = np.loadtxt(u_model_running_time_path)
u_result = np.loadtxt(u_result_path)

with open(a_result_path, 'r') as file:
    new_file_content = file.readlines()
data = []
for line in new_file_content:
    start = line.find('[[') + 2
    end = line.find(']]', start)
    value_str = line[start:end]
    values = [float(val.strip()) for val in value_str.split(',')]
    data.extend(values)
a_result = np.array(data)



test_name = ['not_con_1','not_con_2','PILS']
test = test_name[sel]

root_path = 'C:/Users/song/Desktop/jetsonBoard_AutoEncoder(lon)/'

if sel == 2:
    origin_data, label = Create_PLIS_Data(root_path)
else:
    origin_data, label = Create_VA_Data(root_path,sel)


label = label - 14
color = 'green'

for i in range(2):
    if i == 0:
        zeros_to_add = np.zeros(15, dtype=u_result.dtype)  # u_result와 동일한 데이터 타입의 0으로 채워진 배열 생성
        u_result = np.concatenate((zeros_to_add, u_result))
        result = u_result
        threshold = 2
        save_fig = u_root_path
    else:
        result = a_result
        threshold = 1990.7229
        save_fig = a_root_path
    fig8, ax8 = plt.subplots()
    timestep_index = np.arange(result.shape[0])
    ax8.plot(timestep_index, result, "-", linewidth=2, c=color)
    ax8.axvline(label, 1, 0.1, color='orange', linestyle='--')
    ax8.set_xlabel("time step", fontsize=14)
    ax8.set_ylabel(r"$\delta(t)$", fontsize=14)
    ax8.axhline(threshold, 1, 0.1, color='red', linestyle='-')

    crossing_indices = np.where(result > threshold)[0]
    if len(crossing_indices) > 0:
        if i ==0:
            if sel == 2:
                crossing_indices = crossing_indices[5:]
            index,first_crossing_index = find_first_consecutive_run(crossing_indices,3)
        ax8.plot(first_crossing_index, result[first_crossing_index], 'o', color='red')
        ax8.annotate(f'x={first_crossing_index}',
                     (first_crossing_index, result[first_crossing_index]),
                     textcoords="offset points", xytext=(-10, -10), ha='center')  # x 값 표기
        if index != 0 and i==0:
            for idx in range(index):
                ax8.plot(crossing_indices[idx], result[crossing_indices[idx]], 'o', color='blue')

    # 가장 긴 레이블의 길이를 구합니다.
    longest_label_len = max(len("Upset:"), len("Detection:"), len("Delay:"))

    # 숫자와 'timestep'을 포함한 부분의 길이를 구합니다. 이는 모든 줄에서 동일할 것입니다.
    number_str_len = max(len(f"{label} timestep"), len(f"{first_crossing_index} timestep"),
                         len(f"{first_crossing_index - label} timestep"))

    # 전체 문자열의 길이를 계산합니다 (왼쪽 레이블의 길이 + 숫자 부분의 길이).
    total_length = longest_label_len + number_str_len

    # 포맷팅을 적용합니다.
    upset_str = f"{'Upset:':<{longest_label_len}}  {label} timestep".ljust(total_length)
    detection_str = f"{'Detection:':<{longest_label_len}}  {first_crossing_index} timestep".ljust(
        total_length)
    delay_str = f"{'Delay:':<{longest_label_len}}  {first_crossing_index - label} timestep".ljust(
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

    plt.tight_layout()
    fig8.savefig(save_fig + f'Unet_{test}.png', dpi=300)
    plt.close()



plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False


fig8, ax8 = plt.subplots()
timestep_index = np.arange(u_all_running_time.shape[0])
ax8.plot(timestep_index, a_all_running_time[15:], "-", linewidth=2, c='skyblue',label = '기존 기법')
ax8.plot(timestep_index, u_all_running_time, "-", linewidth=2, c='red',label = '제안 기법')
ax8.set_xlabel("time step", fontsize=14)
ax8.set_ylabel('sec', fontsize=14)

unet_str_all = f"U-Net  mean: {np.mean(u_all_running_time):.3f} min: {np.min(u_all_running_time):.3f} max: {np.max(u_all_running_time):.3f}"
auto_str_all = f"AutoEncoder  mean: {np.mean(a_all_running_time[15:]):.3f} min: {np.min(a_all_running_time[15:]):.3f} max: {np.max(a_all_running_time[15:]):.3f}"
#ax8.text(0.03, 0.97, unet_str, transform=ax8.transAxes, fontsize=14, fontweight='bold',va='top',ha='left', color='red')
#ax8.text(0.03, 0.92, auto_str, transform=ax8.transAxes, fontsize=14, fontweight='bold',va='top',ha='left', color='skyblue')
plt.legend()
plt.tight_layout()
fig8.savefig(u_root_path + f'all_processing_{test}.png', dpi=300)
plt.close()

fig8, ax8 = plt.subplots()
timestep_index = np.arange(u_model_running_time.shape[0])
ax8.plot(timestep_index, a_model_running_time[15:], "-", linewidth=2, c='skyblue',label = '기존 기법')
ax8.plot(timestep_index, u_model_running_time, "-", linewidth=2, c='red',label = '제안 기법')
ax8.set_xlabel("time step", fontsize=14)
ax8.set_ylabel('sec', fontsize=14)

unet_str = f"U-Net mean: {np.mean(u_model_running_time):.3f} min: {np.min(u_model_running_time):.3f} max: {np.max(u_model_running_time):.3f}"
auto_str = f"AutoEncoder  mean: {np.mean(a_model_running_time[15:]):.3f} min: {np.min(a_model_running_time[15:]):.3f} max: {np.max(a_model_running_time[15:]):.3f}"
#ax8.text(0.03, 0.97, unet_str, transform=ax8.transAxes, fontsize=14, fontweight='bold',va='top',ha='left', color='red')
#ax8.text(0.03, 0.92, auto_str, transform=ax8.transAxes, fontsize=14, fontweight='bold',va='top',ha='left', color='skyblue')
plt.legend()
plt.tight_layout()
fig8.savefig(u_root_path + f'model_processing_{test}.png', dpi=300)
plt.close()


