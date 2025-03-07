import numpy as np
import pandas as pd
import h5py,os
from keras.models import load_model

# 항우연
# valiAEMean = [0.00224424, 0.00194687, 0.00241044]
# Inv_Cov = np.array([[324322.14441948, -41443.96395035, -14830.55169333],
#                     [-41443.96395035, 311693.02121456, -82298.09629227],
#                     [-14830.55169333, -82298.09629227, 307579.18892826]])

#내가 구한 것 (not except)
# valiAEMean = [0.00032736, 0.00042464, 0.00020254]
# Inv_Cov = np.array([[10894227.3040371 , -1852964.81081921, -2663715.95210474],
#                     [-1852964.81081921,  6992086.16355535, -2536576.936831  ],
#                     [-2663715.95210474, -2536576.936831  ,  7843481.51137087]])


#내가 구한 것 (except)
# valiAEMean = [0.00044733, 0.00038775, 0.00040261]
# Inv_Cov = np.array([[ 2561959.79612788, -1938089.37668615,  -984096.5558619 ],
#                     [-1938089.37668615,  6560530.00023759,  -591933.27792832],
#                     [ -984096.5558619 ,  -591933.27792832,  3748786.64252654]])


root_path = 'F:/SMJ/2023.08.08(Unet)/'


hdf_path = root_path + f'hdf_file/1D conv_Unet_idx2_(종_test).hdf5'
model_path = 'F:/SMJ/2023.08.08(Unet)/1D conv_Unet_idx2_(lon_train)/seed_4_LR_8e-05_batch size_16/model_check_point/495-0.00030393.hdf5'


sum = 0
all_data = np.empty((0, 15, 3), dtype=np.float32)
hf = h5py.File(hdf_path, 'r')
for initV in range(15, 24, 3):
    for cmdV in range(5, 16, 5):
        for cmdPhi in range(-45, 60, 15):
            for index in range(1, 2):
                key_name = "L_origin_H300_initV" + str(initV) + "_cmdV" + str(cmdV) + "_initPhi0_cmdPhi" + str(cmdPhi) + "_" + str(index) + "_30Hz_data"
                input_data = hf.get(key_name)[:,:,:-1]
                label = hf.get(key_name)[:, -1, -1]
                indices = np.where(label == 0)[0]
                stall_point = indices[0] if indices.size > 0 else None
                input_data = input_data[:stall_point,:,:]
                print(input_data.shape[0])
                sum = sum + input_data.shape[0]
                all_data = np.append(all_data, input_data, axis=0)



model = load_model(model_path)

predict = model.predict(all_data, batch_size=16)
print("predict done")

X = np.abs(predict - all_data)

mu = np.mean(X[:,14,:], axis=0)
centered_data = X[:,14,:] - mu
cov_matrix = np.cov(centered_data, rowvar=False, bias=True)
inv_cov_matrix = np.linalg.inv(cov_matrix)

