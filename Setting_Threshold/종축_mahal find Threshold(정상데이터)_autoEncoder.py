from keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Multiply, Activation, Add,concatenate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib,h5py,os,timeit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

def diff_value(data,n):
    diff_array = np.zeros(len(data))
    for i in range(len(data)):
        if i < n:
            diff_array[i] = 0
        else:
            diff_array[i] = data[i] - data[i-n]
    return diff_array

#항우연(내가 한것)
a_valiAEMean =  [0.00230415, 0.00211229, 0.00265819]
a_Inv_Cov = np.array([[246914.78779075, -66275.16419005, -25370.19992781],
       [-66275.16419005, 213602.62641208, -77047.16127855],
       [-25370.19992781, -77047.16127855, 238461.54595365]])

like_unet = 0

autoencoder_model_path = 'F:/SMJ/2023.08.09(autoencoder)/1D conv_Unet_idx2/seed_4_LR_0.0001_batch size_32/model_check_point/88-0.00202166.hdf5'
autoencoder_model = load_model(autoencoder_model_path)

n = 15
diff = 1

Threshold_save_dir = f'F:/SMJ/2023.10.31(논문 그림 모음)/Threshold/Autoencoder/'
os.makedirs(Threshold_save_dir, exist_ok=True)

# Case
Case2 = ["L_origin_H300_initV15_cmdV5_initPhi0_cmdPhi30_1_30Hz_data",
         "L_origin_H300_initV15_cmdV5_initPhi0_cmdPhi-30_1_30Hz_data",
         "L_origin_H300_initV15_cmdV10_initPhi0_cmdPhi-30_1_30Hz_data",
         "L_origin_H300_initV15_cmdV10_initPhi0_cmdPhi45_1_30Hz_data"]
Case3 = ["L_origin_H300_initV15_cmdV10_initPhi0_cmdPhi30_1_30Hz_data",
         "L_origin_H300_initV15_cmdV10_initPhi0_cmdPhi-45_1_30Hz_data"]
Threshold = 0
results = []
# 입력데이터
hf = h5py.File('F:/SMJ/2023.08.09(autoencoder)/hdf_file/for Threshold(정상).hdf5', 'r')
print(diff)


fig3, ax3 = plt.subplots()
for initV in range(15, 24, 3):
    for cmdV in range(5, 15, 5):
        for cmdPhi in range(-45, 60, 15):
            for index in range(1, 2):
                key_name = "L_origin_H300_initV" + str(initV) + "_cmdV" + str(cmdV) + "_initPhi0_cmdPhi" + str(cmdPhi) + "_" + str(index) + "_30Hz_data"
                print(key_name)

                input_data = hf.get(key_name)[:, :, :-1]

                a_o = autoencoder_model.predict(input_data, batch_size=32)

                a_Recon = a_o - input_data
                a_AA = np.abs(a_Recon - a_valiAEMean)
                a_mahal = np.zeros(a_AA.shape[0])
                # 마할라노비스 계산
                for i in range(a_mahal.shape[0]):
                    a_mahal[i] = np.dot(np.dot(a_AA[i][-1][:], a_Inv_Cov), a_AA[i][-1][:].T)

                max_value = np.max(a_mahal)
                if like_unet == 1:
                    a_mahal = np.sqrt(a_mahal)
                    #MA 및 diff 적용
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
                    #정상 구간 max 값
                    max_value = np.max(moving_avg)

                #plot
                ax3.plot(a_mahal)
                #max 값 저장
                if key_name in Case3:
                    results.append(('Case3',key_name, max_value))
                if key_name in Case2:
                    results.append(('Case2',key_name, max_value))
                else:
                    results.append(('Case1',key_name, max_value))
                #threshold 선정
                if Threshold < max_value:
                    Threshold = max_value

ax3.axhline(Threshold, 1, 0.1, color='red', linestyle='-',lw = 3)
fig3.savefig(Threshold_save_dir + f'threshold_2.png', dpi=300) if like_unet == 1 else fig3.savefig(Threshold_save_dir + f'threshold.png', dpi=300)
plt.close()

sorted_results = sorted(results, key=lambda x: x[2])


txt_name =  "Threshold_2.txt" if like_unet == 1 else "Threshold.txt"

with open(Threshold_save_dir + txt_name, "w") as f:
    for case,key_name, max_value in sorted_results:
        f.write(f"{case}: {key_name} -> {max_value}\n")








# 데이터 추출
values = [item[2] for item in sorted_results]

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
