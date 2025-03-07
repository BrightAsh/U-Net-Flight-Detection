from keras.models import load_model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib,h5py,os,timeit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.io

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
        root_path = path + "data/T-2_long_150Hz.mat";
    else:
        root_path = path + "data/L_origin_H300_initV20_cmdV11_initPhi0_cmdPhi32_150Hz_Cut.mat";

    mat_data = scipy.io.loadmat(root_path)
    flight_data = mat_data['origin_data']
    lon_data = flight_data[:,[17,2,5,49]]
    origin_data = lon_data[0:-1:5,:]
    indices = np.where(origin_data[:,-1] == 0)[0]
    stall_point = indices[0] if indices.size > 0 else None
    origin_data = origin_data[:,:-1]
    return origin_data, stall_point


sel = 0

test_name = ['not_con_1','not_con_2','PILS']
test = test_name[sel]

root_path = 'C:/Users/song/Desktop/jetsonBoard_UNet(lon)/'
Unet_model_path = root_path + 'model/495-0.00030393.hdf5'

if sel == 2:
    origin_data, label = Create_PLIS_Data(root_path)
else:
    origin_data, label = Create_VA_Data(root_path,sel)

save_fig = root_path + f'result/{test}/'

u_valiAEMean = [0.00032736, 0.00042464, 0.00020254]
u_Inv_Cov = np.array([[10894227.3040371 , -1852964.81081921, -2663715.95210474],
                    [-1852964.81081921,  6992086.16355535, -2536576.936831  ],
                    [-2663715.95210474, -2536576.936831  ,  7843481.51137087]])
data_max = np.array([0.2971319307163822, 1.779272024536539, 0.2999026597116440])
data_min = np.array([0.04004381033017002, -0.4247184501683437, - 0.2141452067224887])



Unet_model = load_model(Unet_model_path)
threshold = 2
stall = 0

stack_data = np.zeros((1,15,3))
mahal_sqrt_stack = np.zeros((15))
mahal_MA_stack = np.zeros((2))

unet_running_time = []
all_running_time = []
result = []

for i in range(origin_data.shape[0]):
    all_start_time = time.time()  # 현재 시간 측정 시작

    input_data = origin_data[i]
    preprocessing_data = (input_data - data_min) / (data_max - data_min)
    stack_data[:, :-1, :] = stack_data[:,1:,:]
    stack_data[:, -1, :] = preprocessing_data
    if i < 14: continue;
    if np.any(stack_data == 0):print(f"{i}번째 에러!!!!!")

    start_time = time.time()  # 현재 시간 측정 시작
    output_data = Unet_model(stack_data)
    end_time = time.time()  # 현재 시간 측정 종료

    u_Recon = output_data - stack_data
    u_AA = np.abs(u_Recon - u_valiAEMean)
    u_mahal = np.dot(np.dot(u_AA[:,-1,:], u_Inv_Cov), u_AA[:,-1,:].T)
    u_mahal_sqrt = np.sqrt(u_mahal)
    mahal_sqrt_stack[:-1] = mahal_sqrt_stack[1:]
    mahal_sqrt_stack[-1] = u_mahal_sqrt
    if i < 28: continue;
    if np.any(mahal_sqrt_stack == 0): print(f"{i}번째 에러!!!!!")
    mahal_MA = np.mean(mahal_sqrt_stack)
    mahal_MA_stack[0] = mahal_MA_stack[1]
    mahal_MA_stack[1] = mahal_MA
    if i < 29: continue;
    if np.any(mahal_MA_stack == 0):
        print(f"{i}번째 에러!!!!!")
    delta = mahal_MA_stack[1] - mahal_MA_stack[0]
    if delta <= 2:stall = 0
    else:stall += 1
    if stall >= 3:print(f"{label},, detection: {i}번째 비정상!!!!!!!!!")

    all_end_time = time.time()  # 현재 시간 측정 종료

    elapsed_time = end_time - start_time  # 경과 시간 계산
    unet_running_time.append(elapsed_time)  # 배열에 저장
    all_elapsed_time = all_end_time - all_start_time  # 경과 시간 계산
    all_running_time.append(all_elapsed_time)  # 배열에 저장

    result.append(delta)


with open(save_fig + "unet_running_time.txt", "w") as file:
    for time in unet_running_time:
        file.write(f"{time}\n")
with open(save_fig + "all_running_time.txt", "w") as file:
    for time in all_running_time:
        file.write(f"{time}\n")
with open(save_fig + "result.txt", "w") as file:
    for time in result:
        file.write(f"{time}\n")



color = 'green'
ylabel = f"Delta(t)"

result = np.array(result)


fig8, ax8 = plt.subplots()
timestep_index = np.arange(result.shape[0])
ax8.plot(timestep_index, result, "-", linewidth=2, c=color)
ax8.axvline(label, 1, 0.1, color='orange', linestyle='--')
ax8.set_xlabel("time step", fontsize=14)
ax8.set_ylabel(ylabel, fontsize=14)
ax8.axhline(threshold, 1, 0.1, color='red', linestyle='-')

crossing_indices = np.where(result > threshold)[0]
if len(crossing_indices) > 0:
    index,first_crossing_index = find_first_consecutive_run(crossing_indices,3)
    ax8.plot(first_crossing_index, result[first_crossing_index], 'o', color='red')
    ax8.annotate(f'x={first_crossing_index}',
                 (first_crossing_index, result[first_crossing_index]),
                 textcoords="offset points", xytext=(-10, -10), ha='center')  # x 값 표기
    if index != 0:
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
fig8.savefig(save_fig + f'Unet_PILS.png', dpi=300)
plt.close()

