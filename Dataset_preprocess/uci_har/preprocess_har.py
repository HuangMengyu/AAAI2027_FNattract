from sklearn.model_selection import train_test_split
import torch
import os
import numpy as np
from tqdm import tqdm

#### adapted from TSTCC script #####

data_dir = 'UCI_HAR_Dataset'
output_dir = '/UCI-HAR'

subject_data_train = np.loadtxt(f'{data_dir}/train/subject_train.txt')
subject_data_test = np.loadtxt(f'{data_dir}/test/subject_test.txt')
# Samples
train_acc_x = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_acc_x_train.txt')
train_acc_y = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_acc_y_train.txt')
train_acc_z = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_acc_z_train.txt')
train_gyro_x = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_gyro_x_train.txt')
train_gyro_y = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_gyro_y_train.txt')
train_gyro_z = np.loadtxt(f'{data_dir}/train/Inertial Signals/body_gyro_z_train.txt')

test_acc_x = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_acc_x_test.txt')
test_acc_y = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_acc_y_test.txt')
test_acc_z = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_acc_z_test.txt')
test_gyro_x = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_gyro_x_test.txt')
test_gyro_y = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_gyro_y_test.txt')
test_gyro_z = np.loadtxt(f'{data_dir}/test/Inertial Signals/body_gyro_z_test.txt')

train_data = np.stack((train_acc_x, train_acc_y, train_acc_z,
                       train_gyro_x, train_gyro_y, train_gyro_z), axis=1)
X_test = np.stack((test_acc_x, test_acc_y, test_acc_z,
                      test_gyro_x, test_gyro_y, test_gyro_z), axis=1)
all_subjects = np.concatenate((subject_data_train, subject_data_test), axis=0)
all_data = np.concatenate((train_data, X_test), axis=0)

print("all_subjects shape:", all_subjects.shape, "all_data shape:", all_data.shape)


# labels
train_labels = np.loadtxt(f'{data_dir}/train/y_train.txt')
train_labels -= np.min(train_labels)
y_test = np.loadtxt(f'{data_dir}/test/y_test.txt')
y_test -= np.min(y_test)

all_labels = np.concatenate((train_labels, y_test), axis=0)
print("all_labels shape:", all_labels.shape)

data_dict = dict()
# process all the data first before saving
for i in tqdm(range(len(all_subjects))):
    subject_id = str(int(all_subjects[i]))
    if len(subject_id) == 1:
        subject_id = '0' + subject_id
    if subject_id not in data_dict.keys():
        data_dict[subject_id] = {'data': [], 'labels': []}
    data_dict[subject_id]['data'].append(all_data[i, :])  # exclude subject column
    data_dict[subject_id]['labels'].append(all_labels[i])

print(data_dict.keys())
# convert lists to numpy arrays
for subject_id in data_dict.keys():
    data_dict[subject_id]['data'] = np.array(data_dict[subject_id]['data'])
    data_dict[subject_id]['labels'] = np.array(data_dict[subject_id]['labels'])

for i in data_dict.keys():
    subject_id = i
    print("Saving subject:", subject_id)
        
    sub_output_dir = os.path.join(output_dir, f'{subject_id}')
    if not os.path.exists(sub_output_dir):
        os.makedirs(sub_output_dir, exist_ok=True)
    
    data_sub_output_dir = os.path.join(sub_output_dir, 'data')
    label_sub_output_dir = os.path.join(sub_output_dir, 'label')
    if not os.path.exists(data_sub_output_dir):
        os.makedirs(data_sub_output_dir, exist_ok=True)
    if not os.path.exists(label_sub_output_dir):
        os.makedirs(label_sub_output_dir, exist_ok=True)
    
    print("Data shape:", data_dict[subject_id]['data'].shape, "Label shape:", data_dict[subject_id]['labels'].shape)
    
    np.save(os.path.join(data_sub_output_dir, f'0.npy'), data_dict[subject_id]['data'])
    np.save(os.path.join(label_sub_output_dir, f'0.npy'), data_dict[subject_id]['labels'])
    
