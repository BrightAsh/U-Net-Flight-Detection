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



def generator(data, label, batch, shuffle=False):
    iter = np.arange(0, len(data) // batch)

    while True:
        if shuffle == True:
            np.random.shuffle(iter)
        for i in iter:
            train_data = data[i*batch: (i+1)*batch]
            train_label = label[i*batch: (i+1)*batch]
            yield train_data, train_label

def find_optimal_model():
    check_point_list = os.listdir(check_point_dir)
    list_np = np.array(check_point_list)
    tmp_list = []
    num_of_check_point = list_np.shape[0]

    for i in range(num_of_check_point):
        tmp = list_np[i].split(sep='-')
        tmp_list.append(tmp)

    max_epoch = 0
    index = 0
    for i in range(num_of_check_point):
        if max_epoch < int(tmp_list[i][0]):
            max_epoch = int(tmp_list[i][0])
            index = i

    return tmp_list[index][0] + '-' + tmp_list[index][1]


root_path = 'F:/SMJ/2023.08.08(Unet)/'

learning_1 = 0  # 1:학습 0: 학습x
sel = 0  # test set 선택 0,1 not 2,3,4 used
hdf_num = 1  #0: except

learning_rate = 0.00008
batch_size = 16
seed = 4

#lr = [0.00009,0.00008,0.00007,0.00006,0.00005,0.00004,0.00003,0.00002,0.00001]
#bs = [16,64,128,256,512]

hdf_list = os.listdir(root_path + 'hdf_file/')
print('\n'.join(hdf_list))
hdf_path = root_path + f'hdf_file/{hdf_list[hdf_num]}'
patience = 30
d_300 = 0  # 1: 300개를 5번 붙임(횡) / 0: 5개를 300번 붙임(종)
time_freq = 1 / 30
epoch = 500
model_path = root_path + f'{hdf_list[hdf_num].replace(".hdf5", "")}/seed_{seed}_LR_{learning_rate}_batch size_{batch_size}/'
test_name_list = ['not_used_condition1_','not_used_condition2_','used_condition1_','used_condition2_','used_condition3_','UT-2_Long_','used_condition4_']
test_name = test_name_list[sel]
initializer = tf.keras.initializers.he_normal(seed=seed)
np.random.seed(seed)

check_point_dir = model_path + 'model_check_point/'
os.makedirs(check_point_dir, exist_ok=True)
check_point_path = check_point_dir + '{epoch:02d}-{val_loss:.8f}.hdf5'
result_dir = model_path + 'Result/'
os.makedirs(result_dir, exist_ok=True)
test_path = result_dir + test_name + '/'
prediction_plot_dir = test_path + 'prediction_result_plot/'
os.makedirs(prediction_plot_dir, exist_ok=True)
initial_weights_hist_dir = test_path + 'initial_weights/'
os.makedirs(initial_weights_hist_dir, exist_ok=True)
end_weights_hist_dir = test_path + 'end_weights/'
os.makedirs(end_weights_hist_dir, exist_ok=True)
rmse_result_path = test_path + 'RMSE_result.txt'
Training_time_path = test_path + 'Training_time.txt'

hf = h5py.File(hdf_path, 'r')
training_data = hf.get('training_data')[:,:,:-1]
validation_data = hf.get('validation_data')[:,:,:-1]
if sel == 6:
    hf2 = h5py.File(root_path + f'hdf_file/1D conv_Unet_idx2_연습용(except+except_test).hdf5', 'r')
    test_data = hf2.get(test_name + 'data')[:, :, :-1]
    label = hf2.get(test_name + 'data')[:, -1, -1]
else:
    test_data = hf.get(test_name + 'data')[:, :, :-1]
    label = hf.get(test_name + 'data')[:, -1, -1]

indices= np.where(label == 0)[0]
stall_point = indices[0] if indices.size > 0 else None
timestep = training_data.shape[1]
features = training_data.shape[2]


inputs = tf.keras.layers.Input(shape=(timestep, features))
# Encoder path
econv1 = tf.keras.layers.Conv1D(filters=48, kernel_size=5, activation='relu', padding='same',
                                kernel_initializer=initializer)(inputs)
econv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same',
                                kernel_initializer=initializer)(econv1)
econv3 = tf.keras.layers.Conv1D(filters=96, kernel_size=3, activation='relu', padding='same',
                                kernel_initializer=initializer)(econv2)
# transform path
convtf = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same',
                                kernel_initializer=initializer)(econv3)
# Decoder path
down_kernel_convtf = tf.keras.layers.Conv1D(filters=96, kernel_size=1, activation='relu', padding='same',
                                            kernel_initializer=initializer)(convtf)
up1 = concatenate([econv3, down_kernel_convtf], axis=-1)
dconv1 = tf.keras.layers.Conv1D(filters=96, kernel_size=3, activation='relu', padding='same',
                                kernel_initializer=initializer)(up1)
down_kernel_dconv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu', padding='same',
                                            kernel_initializer=initializer)(dconv1)
up2 = concatenate([econv2, down_kernel_dconv1], axis=-1)
dconv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same',
                                kernel_initializer=initializer)(up2)
down_kernel_dconv2 = tf.keras.layers.Conv1D(filters=48, kernel_size=1, activation='relu', padding='same',
                                            kernel_initializer=initializer)(dconv2)
up3 = concatenate([econv1, down_kernel_dconv2], axis=-1)
dconv3 = tf.keras.layers.Conv1D(filters=48, kernel_size=5, activation='relu', padding='same',
                                kernel_initializer=initializer)(up3)
# 출력 레이어
output = tf.keras.layers.Conv1D(filters=3, kernel_size=1, activation='sigmoid',
                                kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed))(dconv3)
# output = tf.keras.layers.Activation(binary_activation)(output_layer)
model = Model(inputs=inputs, outputs=output)
plot_model(model, to_file=model_path + 'model.png')

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])  test label 설정시

cb_checkpoint = ModelCheckpoint(filepath=check_point_path, monitor='mae', verbose=1, save_best_only=True)
callback_list = [cb_checkpoint]

# 모델 학습 시작
start = timeit.default_timer()
history = model.fit(generator(training_data, training_data, batch_size, True),
                    # batch_size = None일 경우 자동으로 batch size가 설정됨 default = 32
                    batch_size=batch_size,
                    epochs=epoch,
                    callbacks=callback_list,  # CustomCallBack(),
                    steps_per_epoch=training_data.shape[0] // batch_size,
                    shuffle=False,
                    validation_data=generator(validation_data, validation_data, batch_size),
                    validation_steps=validation_data.shape[0] // batch_size)
end = timeit.default_timer()
print("\n Training Time : %.4f" % (end - start))
np.save(result_dir + 'history.npy', history.history)

"""
#가장 val_loss가 적은 check point model을 불러오기
"""
optimal_model_path = find_optimal_model()
model = load_model(check_point_dir + optimal_model_path)
print("Load model: "+ optimal_model_path)

# prediction time calculation
start = timeit.default_timer()
predict = model.predict(test_data, batch_size= batch_size)
end = timeit.default_timer()
print("\n Prediction Time(total) : %.4f" % (end - start))

np.savetxt(test_path + 'Unet_value_prediction_data.csv', predict[:,-1,:], delimiter=",")

with h5py.File(test_path + 'Unet_value_prediction_data.hdf5', 'a') as f:
    # 해당 이름의 데이터셋이 이미 존재하는 경우 삭제
    if test_name in f:
        del f[test_name]
    # 새로운 데이터셋 생성
    f.create_dataset(test_name, data=predict)

# acc & loss 그래프 plotting
data = np.load(result_dir + 'history.npy', allow_pickle=True).item()
training_loss = data['loss']
val_loss = data['val_loss']
epoch = np.arange(len(val_loss))

fig1, ax1 = plt.subplots()
ax1.plot(epoch, val_loss, "-", c="blue", markersize=2, label='val_loss')
ax1.plot(epoch, training_loss, "-", c="green", markersize=2, label='train_loss')
ax1.set_xlabel('epoch', fontsize=14)
ax1.set_ylabel('Loss(MAE)', fontsize=14)
fig1.legend(loc=(0.7, 0.6), fontsize=10)
plt.tight_layout()

fig1.savefig(result_dir + 'Loss_plot.png', dpi=300)
plt.close()

#data_max = np.array([0.2973, 1.7793, 0.2999])
#data_min = np.array([0.0423, -0.5328, -0.2599])

data_max = [0.2971, 1.7793, 0.2999]
data_min = [0.0400, -0.4247, -0.2141]

predict_cr = predict[:,-1,:]

#stall_point = int(0.2/time_freq)

alpha_predict = (predict_cr[:, 0]*(data_max[0] - data_min[0]) + data_min[0]) * 57.2974
q_predict = (predict_cr[:, 1]*(data_max[1] - data_min[1]) + data_min[1]) * 57.2974
theta_predict = (predict_cr[:, 2]*(data_max[2] - data_min[2]) + data_min[2]) * 57.2974

test_label = test_data[:,-1,:]

alpha_true = (test_label[:, 0]*(data_max[0] - data_min[0]) + data_min[0]) * 57.2974
q_true = (test_label[:, 1]*(data_max[1] - data_min[1]) + data_min[1]) * 57.2974
theta_true = (test_label[:, 2]*(data_max[2] - data_min[2]) + data_min[2]) * 57.2974



fig2, ax2 = plt.subplots()
#timestep_index = np.arange(len(predict[:end_index, 0]))
timestep_index = np.arange(len(predict_cr[:, 0]))
ax2.plot(timestep_index, alpha_true, "-", linewidth = 2, c="black", label='True alpha')
ax2.plot(timestep_index, alpha_predict, ":", linewidth = 3, c="red", label='CNN+LSTM alpha prediction')
ax2.set_xlabel("time step", fontsize=14)
ax2.set_ylabel("alpha (deg)", fontsize=14)
ax2.axvline(stall_point, 1, 0.1, color='orange', linestyle='--') if stall_point is not None else None
#plt.ylim(5,15)
plt.tight_layout()
fig2.savefig(prediction_plot_dir + 'alpha_prediction_plot.png', dpi=300)
plt.close()


fig3, ax3 = plt.subplots()
timestep_index = np.arange(len(predict_cr[:, 1]))
ax3.plot(timestep_index, q_true, "-", linewidth = 2, c="black", label='True q')
ax3.plot(timestep_index, q_predict, ":", linewidth = 3, c="red", label='CNN+LSTM q prediction')
ax3.set_xlabel("time step", fontsize=14)
ax3.set_ylabel("q (deg/sec)", fontsize=14)
ax3.axvline(stall_point, 1, 0.1, color='orange', linestyle='--') if stall_point is not None else None
#plt.ylim(6,26)
plt.tight_layout()

fig3.savefig(prediction_plot_dir + 'q_prediction_plot.png', dpi=300)
plt.close()

fig4, ax4 = plt.subplots()
timestep_index = np.arange(len(predict_cr[:, 2]))
ax4.plot(timestep_index, theta_true, "-", linewidth = 2, c="black", label='True theta')
ax4.plot(timestep_index, theta_predict, ":", linewidth = 3, c="red", label='CNN+LSTM theta prediction')
ax4.set_xlabel("time step", fontsize=14)
ax4.set_ylabel("theta (deg)", fontsize=14)
ax4.axvline(stall_point, 1, 0.1, color='orange', linestyle='--') if stall_point is not None else None
plt.tight_layout()

fig4.savefig(prediction_plot_dir + 'theta_prediction_plot.png', dpi=300)
plt.close()

#upset 이후 데이터 개수
upset_step = len(label) - stall_point if stall_point is not None else 1
#upset이전 predict, true값
alpha_predict_upset = (predict_cr[:-upset_step, 0]*(data_max[0] - data_min[0]) + data_min[0]) * 57.2974
q_predict_upset = (predict_cr[:-upset_step, 1]*(data_max[1] - data_min[1]) + data_min[1]) * 57.2974
theta_predict_upset = (predict_cr[:-upset_step, 2]*(data_max[2] - data_min[2]) + data_min[2]) * 57.2974

alpha_true_upset = (test_label[:-upset_step, 0]*(data_max[0] - data_min[0]) + data_min[0]) * 57.2974
q_true_upset = (test_label[:-upset_step, 1]*(data_max[1] - data_min[1]) + data_min[1]) * 57.2974
theta_true_upset = (test_label[:-upset_step, 2]*(data_max[2] - data_min[2]) + data_min[2]) * 57.2974

print("CNN+LSTM alpha rmse:" + str(np.sqrt(np.mean((alpha_true_upset - alpha_predict_upset) ** 2))))
print("CNN+LSTM q rmse:" + str(np.sqrt(np.mean((q_true_upset - q_predict_upset) ** 2))))
print("CNN+LSTM theta rmse:" + str(np.sqrt(np.mean((theta_true_upset - theta_predict_upset) ** 2))))

print("alpha maximum error: " + str(np.sqrt(np.max((alpha_true_upset - alpha_predict_upset) ** 2))))
print("q maximum error: " + str(np.sqrt(np.max((q_true_upset - q_predict_upset) ** 2))))
print("theta maximum error: " + str(np.sqrt(np.max((theta_true_upset - theta_predict_upset) ** 2))))

if not os.path.exists(rmse_result_path):
    with open(rmse_result_path, 'w') as f:
        f.write(" Prediction Time(total) : %.4f" % (end - start))
        f.write("\n\nCNN+LSTM alpha rmse:" + str(np.sqrt(np.mean((alpha_true_upset - alpha_predict_upset) ** 2))))
        f.write("\nCNN+LSTM q rmse:" + str(np.sqrt(np.mean((q_true_upset - q_predict_upset) ** 2))))
        f.write("\nCNN+LSTM theta rmse:" + str(np.sqrt(np.mean((theta_true_upset - theta_predict_upset) ** 2))))
        f.write("\n\nalpha maximum error: " + str(np.sqrt(np.max((alpha_true_upset - alpha_predict_upset) ** 2))))
        f.write("\nq maximum error: " + str(np.sqrt(np.max((q_true_upset - q_predict_upset) ** 2))))
        f.write("\ntheta maximum error: " + str(np.sqrt(np.max((theta_true_upset - theta_predict_upset) ** 2))))
