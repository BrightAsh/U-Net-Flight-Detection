from keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Multiply, Activation, Add,concatenate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib,h5py,os,timeit
matplotlib.use('Agg')
from tensorflow.keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumBarunGothic'


def diff_value(data,n):
    diff_array = np.zeros(len(data))
    for i in range(len(data)):
        if i < n:
            diff_array[i] = 0
        else:
            diff_array[i] = data[i] - data[i-n]
    return diff_array

if True:
    color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    color_dict = {}

    color_index = 0
    for initV in range(15, 24, 3):
        for cmdV in range(5, 15, 5):
            for cmdPhi in range(-45, 60, 15):
                for index in range(1, 2):
                    key_name = "L_origin_H300_initV" + str(initV) + "_cmdV" + str(cmdV) + "_initPhi0_cmdPhi" + str(cmdPhi) + "_" + str(index) + "_30Hz_data"
                    color_dict[key_name] = color_list[color_index % len(color_list)]
                    color_index += 1



#내가 구한 것 (not except)
u_valiAEMean = [0.00032736, 0.00042464, 0.00020254]
u_Inv_Cov = np.array([[10894227.3040371 , -1852964.81081921, -2663715.95210474],
                    [-1852964.81081921,  6992086.16355535, -2536576.936831  ],
                    [-2663715.95210474, -2536576.936831  ,  7843481.51137087]])


Unet_model_path = 'F:/SMJ/2023.08.08(Unet)/1D conv_Unet_idx2_(lon_train)/seed_4_LR_8e-05_batch size_16/model_check_point/495-0.00030393.hdf5'
Unet_model= load_model(Unet_model_path)
hf = h5py.File('F:/SMJ/2023.08.08(Unet)/hdf_file/1D conv_Unet_idx2_(종_test).hdf5', 'r')

Threshold_save_dir = 'F:/SMJ/2023.10.31(논문 그림 모음)/종축/Threshold/Unet/'
#os.makedirs(Unet_threshold_path, exist_ok=True)


n = 15
diff = 1

Threshold = 0

results = []

each_sinario_threshold_list = []
each_sinario_threshold_key_list = []

fig3, ax3 = plt.subplots()

for initV in range(15, 24, 3):
    for cmdV in range(5, 15, 5):
        for cmdPhi in range(-45, 60, 15):
            for index in range(1, 2):
                key_name = "L_origin_H300_initV" + str(initV) + "_cmdV" + str(cmdV) + "_initPhi0_cmdPhi" + str(cmdPhi) + "_" + str(index) + "_30Hz_data"
                #그래프 색 맞추기
                plot_color = color_dict.get(key_name, "black")
                input_data = hf.get(key_name)[:,:,:-1]

                label = hf.get(key_name)[:, -1, -1]
                indices = np.where(label == 0)[0]
                stall_point = indices[0] if indices.size > 0 else None


                u_o = Unet_model.predict(input_data, batch_size= 16)
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

                max_value = np.max(moving_avg[:stall_point])
                ax3.plot(moving_avg[:stall_point])
                results.append((key_name, max_value))

                if Threshold < max_value:
                    Threshold = max_value


ax3.axhline(Threshold, 1, 0.1, color='red', linestyle='-',lw = 3)
fig3.savefig(Threshold_save_dir + f'threshold.png', dpi=300)
plt.close()

sorted_results = sorted(results, key=lambda x: x[1])

with open(Threshold_save_dir+"Threshold.txt", "w") as f:
    for key_name, max_value in sorted_results:
        f.write(f" {key_name} -> {max_value}\n")


# 데이터 추출
values = [item[1] for item in sorted_results]

bins = 50

# 히스토그램 생성
hist_values, bin_edges, patches = plt.hist(values, bins=bins, edgecolor='black', alpha=0.7)

colors = ['red', 'orange', 'yellow', 'green', 'skyblue', 'coral' ,'olive', 'teal', 'mediumvioletred', 'cadetblue', 'darkseagreen', 'mediumpurple', 'goldenrod']
repeat_colors = colors * (len(hist_values) // len(colors)) + colors[:len(hist_values) % len(colors)]

i = 0
for value, patch in zip(hist_values, patches):
    if value != 0:
        patch.set_facecolor(repeat_colors[i])
        plt.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.5, int(value),
                 ha='center', va='bottom')
        i += 1

# 텍스트 추가
min_val = min(values)
max_val = max(values)
each_range = (max_val - min_val) / bins

text_position_x = 0.7 * plt.xlim()[1]
text_position_y = 0.9 * plt.ylim()[1]
plt.text(text_position_x, text_position_y, f'min: {min_val:.2f}', ha='left')
plt.text(text_position_x, text_position_y - 0.05 * plt.ylim()[1], f'max: {max_val:.2f}', ha='left')
plt.text(text_position_x, text_position_y - 0.1 * plt.ylim()[1], f'range: {each_range:.2f}', ha='left')

plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.tight_layout()

# 그래프 저장하기
plt.savefig(Threshold_save_dir + 'histogram.png', dpi=300)
plt.close()
