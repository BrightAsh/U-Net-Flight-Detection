import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import h5py
import copy

#training data 전처리
def training_data_preprocessing(file_path, hdf_path, timestep, feature,first=False):
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)
    data_tmp = np.loadtxt(file_path, delimiter=',', dtype='float')

    zero_start_index = np.where(data_tmp[:, -1] == 0)[0][0] if np.any(data_tmp[:, -1] == 0) else data_tmp.shape[0]
    data_tmp = data_tmp[:zero_start_index, :]



    total_length = np.shape(data_tmp)[0]
    numOfData = int(total_length - timestep + 1)
    # window sliding 입력 데이터를 저장할 임시 변수, data size = (전체 데이터 수, time_step, feature)
    window_sliding_data = np.zeros(numOfData * timestep * feature).reshape(numOfData, timestep, feature)
    # 입력에 대해 모델이 예측해야할 정답 데이터를 저장할 임시 변수 data size = (전체 데이터 수, 예측 변수 수)
    window_sliding_prediction = copy.deepcopy(window_sliding_data)



    i = 0
    #[q theta alpha ]
    # 0  1   2
    while i < numOfData:
        window_sliding_data[i, :, 0] = data_tmp[i:i + timestep, 2]
        window_sliding_data[i, :, 1] = data_tmp[i:i + timestep, 0]
        window_sliding_data[i, :, 2] = data_tmp[i:i + timestep ,1]
        window_sliding_data[i, :, 3] = data_tmp[i:i + timestep, 3]
        #print(f'{i}번째 데이터 처리중')
        i = i + 1

    training_data, validation_data, training_prediction, validation_prediction = train_test_split(window_sliding_data, window_sliding_prediction,test_size=0.3, random_state=seed)

    if first == True:
        with h5py.File(hdf_path, "w") as f:
            f.create_dataset("training_data", data=training_data[:], maxshape=(None, timestep, feature))
            f.create_dataset("validation_data", data=validation_data[:], maxshape=(None, timestep, feature))
    else:
        with h5py.File(hdf_path, "r+") as f:
            f["training_data"].resize((f["training_data"].shape[0] + training_data.shape[0]), axis=0)
            f["training_data"][-training_data.shape[0]:] = training_data[:]

            f["validation_data"].resize((f["validation_data"].shape[0] + validation_data.shape[0]), axis=0)
            f["validation_data"][-validation_data.shape[0]:] = validation_data[:]
    return training_data.shape[0]
#test data 전처리
def test_data_preprocessing(file_path, hdf_path, timestep, feature, key_name, first=False):
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)
    data_tmp = np.loadtxt(file_path, delimiter=',', dtype='float')
    total_length = np.shape(data_tmp)[0]
    numOfData = int(total_length - timestep + 1)
    window_sliding_data = np.zeros(numOfData * timestep * feature).reshape(numOfData, timestep, feature)
    i = 0
    while i < numOfData:
        window_sliding_data[i, :, 0] = data_tmp[i:i + timestep, 2]
        window_sliding_data[i, :, 1] = data_tmp[i:i + timestep, 0]
        window_sliding_data[i, :, 2] = data_tmp[i:i + timestep, 1]
        window_sliding_data[i, :, 3] = data_tmp[i:i + timestep, 3]
        i = i + 1
    if first == True:
        with h5py.File(hdf_path, "w") as f:
            data_key = key_name+'_data'
            f.create_dataset(data_key, data=window_sliding_data[:])
    else:
        with h5py.File(hdf_path, "r+") as f:
            data_key = key_name+'_data'
            f.create_dataset(data_key, data=window_sliding_data[:])


sel = 1

hdf_path = 'D:/deeplearning/Unet_final/data preprocessing/2 pycharm data/train.hdf5' if sel ==1 else 'D:/deeplearning/Unet_final/data preprocessing/2 pycharm data/test.hdf5'

data_dir = 'D:/deeplearning/Unet_final/data preprocessing/1 matlab data/'
training_data_dir_path = data_dir + 'training_data/종/'
test_dir_path = data_dir + "test_data/종/"

timestep = 15
feature = 4

total_training_data_size = 0
i = 0
if sel == 1:
    for initV in range(15, 24, 3):
        for cmdV in range(5, 20, 5):
            for cmdPhi in range(-45, 60, 15):
                # csv file index (down sampling index)
                for idx in range(1,6):
                    file_name = "L_origin_H300_initV"+str(initV)+"_cmdV"+str(cmdV)+"_initPhi0_cmdPhi"+str(cmdPhi)+"_"+str(idx)+"_30Hz.csv"
                    file_path = training_data_dir_path + file_name

                    if i == 0:
                        total_training_data_size += training_data_preprocessing(file_path, hdf_path, timestep, feature, first=True)
                    else:
                        total_training_data_size += training_data_preprocessing(file_path, hdf_path, timestep, feature, first=False)
                    i = i+1

i = 0
if sel == 2:
    for initV in range(15, 24, 3):
        for cmdV in range(5, 20, 5):
            for cmdPhi in range(-45, 60, 15):
                file_name = "L_origin_H300_initV"+str(initV)+"_cmdV"+str(cmdV)+"_initPhi0_cmdPhi"+str(cmdPhi)+"_"+str('1')+"_30Hz.csv"
                file_path = training_data_dir_path + file_name

                if i == 0:
                    test_data_preprocessing(file_path, hdf_path, timestep, feature,file_name, first=True)
                else:
                    test_data_preprocessing(file_path, hdf_path, timestep, feature,file_name, first=False)
                i = i+1

    file_name = "L_origin_H300_initV17_cmdV13_initPhi0_cmdPhi-32_150Hz_Cut.csv"
    file_path = test_dir_path + file_name
    test_data_preprocessing(file_path, hdf_path, timestep, feature, 'UT-1',first=False)

    file_name = "T-2_long_150Hz.csv"
    file_path = test_dir_path + file_name
    test_data_preprocessing(file_path, hdf_path, timestep, feature,'UT-2',first=False)

    file_name = "L_origin_H300_initV20_cmdV11_initPhi0_cmdPhi32_150Hz_Cut.csv"
    file_path = test_dir_path + file_name
    test_data_preprocessing(file_path, hdf_path, timestep, feature, 'UT-3',first=False)

    file_name = "L_origin_H300_initV15_cmdV10_initPhi0_cmdPhi30_150Hz_Long.csv"
    file_path = test_dir_path + file_name
    test_data_preprocessing(file_path, hdf_path, timestep, feature, 'used_condition1',first=False)

