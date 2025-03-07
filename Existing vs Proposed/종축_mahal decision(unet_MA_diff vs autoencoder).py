import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib
matplotlib.use('Agg')


plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False




u_MA_save_dir = f'F:/SMJ/2023.10.31(논문 그림 모음)/'
#unet_MA15_diff_1_3th_detection = [ 6, 4, 3, 4, 4, 21, 15, 15, 2, 3, 3, 4, 3, 3, 3, 3, 4, 4, 5, 5, 4, 4, 7, 1, 1, 5, 5, 3, 3, 20, 1, 1, 4, 4, 2, 2, 3, 3, 4, 0,36, 5, 52]
#autoencoder =                    [19,15,15, 4, 4, 27, 24, 24, 4, 5, 5, 7, 6, 6, 5, 5, 8, 8,13,13, 7, 7,10, 2, 2, 5, 5, 6, 6, 26, 2, 2, 4, 4, 9, 9, 4, 5,13, 0,34, 5, 61]

#autoencoder = [4,5,15,19,15,4,4,13,24,27,24,69,0,5,6,5,4,5,7,6,7,13,8,5,6,13,7,6,5,2,10,2,5,6,9,4,2,26,2,4,9,34,5]
#unet_MA15_diff_1_3th_detection = [4,3,3,6,4,3,4,4,15,21,15,59,0,3,3,3,2,3,4,3,4,5,4,3,4,5,4,3,5,1,7,1,5,3,2,4,1,20,1,4,2,36,5]

#autoencoder =                    [4,5,6,12,7,4,4,7,16,21,16,69,0,5,5,3,3,4,5,5,6,8,4,5,5,13,6,5,5,2,11,2,5,5,4,4,2,20,2,4,4,34,5]
#unet_MA15_diff_1_3th_detection = [4,3,3, 6,4,3,4,4,15,21,15,59,0,3,3,3,2,3,4,3,4,5,4,3,4,5,4,3,5,1,7,1,5,3,2,4,1,  20,1,4,2,36,5]

autoencoder = [4,5,15,19,16,5,5,14,21,28,25,69,0,6,6,5,4,5,7,6,7,13,6,6,7,13,7,6,5,2,11,2,5,6,7,5,3,27,3,5,7,34,5]
unet_MA15_diff_1_3th_detection = [4,3,3, 6,4,3,4,4,15,21,15,59,0,3,3,3,2,3,4,3,4,5,4,3,4,5,4,3,5,1,7,1,5,3,2,4,1,  20,1,4,2,36,5]

unet_freq = Counter(unet_MA15_diff_1_3th_detection)
autoencoder_freq = Counter(autoencoder)

x = np.arange(71)
bar_width = 0.35
# unet의 빈도를 빨간 막대로 표시
y_unet = [unet_freq[i] for i in x]
plt.bar(x - bar_width/2, y_unet, bar_width, color='red', label='제안하는 탐지 기법') # 위치 조절

# autoencoder의 빈도를 파란 막대로 표시
y_autoencoder = [autoencoder_freq[i] for i in x]
plt.bar(x + bar_width/2, y_autoencoder, bar_width, color='skyblue', label='기존 탐지 기법') # 위치 조절

plt.legend()
plt.xlabel('Delay Timestep')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(u_MA_save_dir + f'막대 unet_MA15_diff_1_3th_detection vs autoencoder.png', dpi=300)
plt.close() # 화면에 그림을 보여줍니다.


###################################################
data1 = autoencoder
data2 = unet_MA15_diff_1_3th_detection

max_val = max(max(data2), max(data1)) + 10
min_val = min(min(data2), min(data1)) - 10
x = np.linspace(min_val, max_val, 1000)
y = x
plt.plot(x, y, color='grey', linewidth=0.5)
plt.fill_between(x, y, min_val - 1, color='yellow')
plt.axhline(0, color='grey', linewidth=0.5)
plt.axvline(0, color='grey', linewidth=0.5)

plt.scatter(data1, data2, color='black',s = 30)
plt.xlabel('기존 탐지 기법')
plt.ylabel('제안하는 탐지 기법')
plt.tight_layout()

plt.savefig(u_MA_save_dir + f'막대 unet_MA15_diff_1_3th_detection vs autoencoder_2.png', dpi=300)
plt.close()


#학습에 사용된것만
# 2auto = [4,5,6,12,7,4,4,7,16,21,16,69,0,5,5,3,3,4,5,5,6,8,4,5,5,13,6,5,5,2,11,2,5,5,4,4,2,20,2,4,4]
auto = [4,5,15,19,16,5,5,14,21,28,25,69,0,6,6,5,4,5,7,6,7,13,6,6,7,13,7,6,5,2,11,2,5,6,7,5,3,27,3,5,7]
# 1 auto = [4,5,15,19,15,4,4,13,24,27,24,69,0,5,6,5,4,5,7,6,7,13,8,5,6,13,7,6,5,2,10,2,5,6,9,4,2,26,2,4,9]
unet = [4,3,3,6,4,3,4,4,15,21,15,59,0,3,3,3,2,3,4,3,4,5,4,3,4,5,4,3,5,1,7,1,5,3,2,4,1,20,1,4,2,27]

############
u = sum(unet)/len(unet)
a = sum(auto)/len(auto)




#autoencoder = [4,5,15,19,15,4,4,13,24,27,24,69,0,5,6,5,4,5,7,6,7,13,8,5,6,13,7,6,5,2,10,2,5,6,9,4,2,26,2,4,9,34,5]
#autoencoder =                    [4,5,6,12,7,4,4,7,16,21,16,69,0,5,5,3,3,4,5,5,6,8,4,5,5,13,6,5,5,2,11,2,5,5,4,4,2,20,2,4,4,34,5]
autoencoder = [4,5,15,19,16,5,5,14,21,28,25,69,0,6,6,5,4,5,7,6,7,13,6,6,7,13,7,6,5,2,11,2,5,6,7,5,3,27,3,5,7,34,5]
unet_MA15_diff_1_3th_detection = [4,3,3, 6,4,3,4,4,15,21,15,59,0,3,3,3,2,3,4,3,4,5,4,3,4,5,4,3,5,1,7,1,5,3,2,4,1,  20,1,4,2,36,5]
u = sum(unet_MA15_diff_1_3th_detection)/len(unet_MA15_diff_1_3th_detection)
a = sum(autoencoder)/len(autoencoder)







a2 = [1,6,5,9,6,5,1,6,9,15,8,54,1,6,5,4,5,4,5,6,7,6,4,5,4,7,7,6,6,3,12,3,6,6,5,5,3,14,3,6,5]





from keras.models import load_model
import tensorflow as tf

Unet_model_path = 'F:/SMJ/2023.08.08(Unet)/1D conv_Unet_idx2_(lon_train)/seed_4_LR_8e-05_batch size_16/model_check_point/495-0.00030393.hdf5'
autoencoder_model_path = 'F:/SMJ/2023.08.09(autoencoder)/1D conv_Unet_idx2/seed_4_LR_0.0001_batch size_32/model_check_point/88-0.00202166.hdf5'
autoEncoder_model = load_model(autoencoder_model_path)
Unet_model = load_model(Unet_model_path)


# 모델 복잡도를 계산하는 함수
def calculate_model_complexity(model):
    return model.count_params()

# 모델 복잡도 계산
complexity1 = calculate_model_complexity(autoEncoder_model)
complexity2 = calculate_model_complexity(Unet_model)

print(f"autoEncoder_model 1 Complexity: {complexity1}")
print(f"Unet_model 2 Complexity: {complexity2}")

# 복잡도 비교
if complexity1 > complexity2:
    print("autoEncoder_model 1 is more complex than Model 2")
else:
    print("Unet_model 2 is more complex than Model 1")


def estimate_flops(model):
    total_flops = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            input_shape = layer.input_shape
            output_shape = layer.output_shape
            kernel_shape = layer.kernel_size[0]
            flops_per_instance = kernel_shape * input_shape[-1] * output_shape[-1]
            total_flops += flops_per_instance * output_shape[1]

        elif isinstance(layer, tf.keras.layers.Bidirectional):
            if isinstance(layer.layer, tf.keras.layers.LSTM):
                lstm_layer = layer.layer
                input_shape = layer.input_shape[-1]
                hidden_size = lstm_layer.units
                flops_per_instance = 4 * (input_shape + hidden_size) * hidden_size + 4 * hidden_size
                total_flops += 2 * flops_per_instance * output_shape[1]  # Bi-directional이므로 2배

    return total_flops


# 모델 정의 후 연산량 추정
flops1 = estimate_flops(autoEncoder_model)
flops2 = estimate_flops(Unet_model)

print(f"Estimated FLOPs for Model 1: {flops1}")
print(f"Estimated FLOPs for Model 2: {flops2}")

# 연산량 비교
if flops1 > flops2:
    print("Model 1 has more computational operations than Model 2")
else:
    print("Model 2 has more computational operations than Model 1")



#  26.267272087293208
#  26.359155990957046