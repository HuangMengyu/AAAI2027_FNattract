import os
import numpy as np
from tqdm import tqdm
import gc
import random

def get_path_loader(parser): # get the data path and label path for each subject
    path = None

    path = []
    for i in range(83):
        if i not in [39, 68, 69, 78, 79]:
            if i < 10:
                path.append(f"0{i}")
            else:
                path.append(f"{i}")

    
    path_name = dict()

    for t_idx in path:
        """Load data files"""
        num = 0
        file_path = parser['filepath'] + f"/{t_idx}/data"
        label_path = parser['filepath'] + f"/{t_idx}/label"
        while os.path.exists(file_path + f"/{num}.npy"):
            if not t_idx in path_name.keys():
                path_name[t_idx] = [[file_path + f"/{num}.npy", label_path + f"/{num}.npy"]]
            else:
                path_name[t_idx].append([file_path + f"/{num}.npy", label_path + f"/{num}.npy"])
            num += 1
    seq_num = 0
    epo_num = 0
    for i in path_name.keys():
        seq_num += len(path_name[i][0])
        epo_num += len(path_name[i][0])*20
    print("total number of sequences:", seq_num)
    print("total number of epoches in sequences:", epo_num)

    return path, path_name


def get_idx(parser, path, dataset_name = 'SleepEDFx', valid_ratio=0.2): # path is the list of indexes 
    print("number of subjects: ", len(path))

    if dataset_name in ['PAMAP2', 'PAMAP2_3']:
        num_val = int(np.round(len(path) * valid_ratio))
    else:
        num_val = int(np.round(len(path) * valid_ratio)) + 1

    if dataset_name == 'SleepEDFx':
        if parser["Fold"] == 1:
            test_idx = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
        elif parser["Fold"] == 2:
            test_idx = ['16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
        elif parser["Fold"] == 3:
            test_idx = ['30', '31', '32', '33', '34', '35', '36', '37', '38', '40', '41', '42', '43', '44', '45', '46']
        elif parser["Fold"] == 4:
            test_idx = ['47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62']
        elif parser["Fold"] == 5:
            test_idx = ['63', '64', '65', '66', '67', '70', '71', '72', '73', '74', '75', '76', '77', '80', '81', '82']
    elif dataset_name == 'SleepEDFx_3':
        if parser["Fold"] == 1:
            test_idx = ['01', '02', '04', '05']
        elif parser["Fold"] == 2:
            test_idx = ['06', '07', '08', '09']
        elif parser["Fold"] == 3:
            test_idx = ['10', '11', '12', '13']
        elif parser["Fold"] == 4:
            test_idx = ['14', '15', '16', '17', '18']
        elif parser["Fold"] == 5:
            test_idx = ['19', '20', '21', '22', '24']
    elif dataset_name in ['PAMAP2', 'PAMAP2_3']:
        # 4 fold cross validation setting
        if parser["Fold"] == 1:
            test_idx = ['101', '102']
        elif parser["Fold"] == 2:
            test_idx = ['103', '104']
        elif parser["Fold"] == 3:
            test_idx = ['105', '106']
        elif parser["Fold"] == 4:
            test_idx = ['107', '108']
    elif dataset_name in ['UCI-HAR', 'UCI-HAR_total']:
        if parser["Fold"] == 1:
            test_idx = ['01', '02', '03', '04', '05', '06']
        elif parser["Fold"] == 2:
            test_idx = ['07', '08', '09', '10', '11', '12']
        elif parser["Fold"] == 3:
            test_idx = ['13', '14', '15', '16', '17', '18']
        elif parser["Fold"] == 4:
            test_idx = ['19', '20', '21', '22', '23', '24']
        elif parser["Fold"] == 5:
            test_idx = ['25', '26', '27', '28', '29', '30']

   

    idx = sorted(list(set(path) - set(test_idx)))
    if dataset_name in ['PAMAP2', 'PAMAP2_3'] and '109' in idx:
        idx.remove('109')  # subject 109 has very less data, so we do not use it for validation
    val_idx = list(np.random.choice(idx, num_val , replace=False))
    
    train_idx = [i for i in idx if i not in val_idx]
    if dataset_name in ['PAMAP2', 'PAMAP2_3'] and '109' not in test_idx and '109' not in val_idx:
        train_idx.append('109') # put back subject 109 for training


    train_idx = [str(i) for i in train_idx]
    val_idx = [str(i) for i in val_idx]
    test_idx = [str(i) for i in test_idx]

    return train_idx, val_idx, test_idx

def get_path_loader_new(input_path, dataset_name='SleepEDFx'): # get the data path and label path for each subject
    path = []
    if dataset_name == 'SleepEDFx':
        for i in range(83):
            if i not in [39, 68, 69, 78, 79]:
                if i < 10:
                    path.append(f"0{i}")
                else:
                    path.append(f"{i}")
    elif dataset_name == 'SleepEDFx_3':
        for i in range(1, 25):
            if i not in [3, 23]:
                if i < 10:
                    path.append(f"0{i}")
                else:
                    path.append(f"{i}")
    elif dataset_name in ['PAMAP2', 'PAMAP2_3']:
        for i in range(101, 110):
            path.append(f"{i}")
    elif dataset_name in ['UCI-HAR', 'UCI-HAR_total']:
        for i in range(1, 31):
            if i < 10:
                path.append(f"0{i}")
            else:
                path.append(f"{i}")

    path_name = dict()  


    for t_idx in path:
        """Load data files"""
        file_path = input_path + f"/{t_idx}/data"
        label_path = input_path + f"/{t_idx}/label"
        files = os.listdir(file_path)
        for i in files:
            if not t_idx in path_name.keys():
                path_name[t_idx] = [[os.path.join(file_path, i), os.path.join(label_path, i)]]
            else:
                path_name[t_idx].append([os.path.join(file_path, i), os.path.join(label_path, i)])

    print(path)
    print(path_name.keys())
    return path, path_name

def get_loader(num, path, path_name, dataset_name = 'SleepEDFx', valid_ratio=0.2):
    train_data = []
    train_label = []
    finetune_train_data = []
    finetune_train_label = []
    finetune_val_data = []
    finetune_val_label = []
    test_data = []
    test_label = []
    train_idx, val_idx, test_idx = get_idx(num, path, dataset_name = dataset_name, valid_ratio=valid_ratio)
    print("Training subjects:", train_idx)
    print("Validation subjects:", val_idx)
    print("Testing subjects:", test_idx)

    # split finetune train and finetune val sets
    total_num = len(val_idx)
    idx = val_idx
    random.shuffle(idx)
    valid_num = np.maximum(int(np.round(total_num * valid_ratio)),1)
    print("Number of finetune validation subjects:", valid_num)
    finetune_val_idx = idx[:valid_num]
    finetune_train_idx = idx[valid_num:]

    for t_idx in tqdm(train_idx):
        assert t_idx in path_name.keys(), f"{t_idx} not in path_name"
        sub_path = path_name[t_idx]
        for i in range(len(sub_path)):
            temp_data = np.load(sub_path[i][0], mmap_mode='r')
            temp_label = np.load(sub_path[i][1], mmap_mode='r')
            train_data.append(temp_data)
            train_label.append(temp_label)

            if len(train_data) % 5 == 0:
                gc.collect()

    train_data = np.concatenate(train_data, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    print("Number of pretrain samples:", train_data.shape)
    assert train_data.shape[0] == train_label.shape[0], "finetune_val data and label size do not match!"

    for v_idx in finetune_train_idx:
        for i in range(len(path_name[v_idx])):
            temp_data = np.load(path_name[v_idx][i][0], mmap_mode='r')
            temp_label = np.load(path_name[v_idx][i][1], mmap_mode='r')
            finetune_train_data.append(temp_data)
            finetune_train_label.append(temp_label)
    finetune_train_data = np.concatenate(finetune_train_data, axis=0)
    finetune_train_label = np.concatenate(finetune_train_label, axis=0)
    assert finetune_train_data.shape[0] == finetune_train_label.shape[0], "finetune_val data and label size do not match!"
    print("Number of finetune training samples:", finetune_train_data.shape)

    for v_idx in finetune_val_idx:
        for i in range(len(path_name[v_idx])):
            temp_data = np.load(path_name[v_idx][i][0], mmap_mode='r')
            temp_label = np.load(path_name[v_idx][i][1], mmap_mode='r')
            finetune_val_data.append(temp_data)
            finetune_val_label.append(temp_label)
    finetune_val_data = np.concatenate(finetune_val_data, axis=0)
    finetune_val_label = np.concatenate(finetune_val_label, axis=0)
    assert finetune_val_data.shape[0] == finetune_val_label.shape[0], "finetune_val data and label size do not match!"
    print("Number of finetune validation samples:", finetune_val_data.shape)

    print(test_idx)
    print(path_name.keys())
    for t_idx in test_idx:
        for i in range(len(path_name[t_idx])):
            temp_data = np.load(path_name[t_idx][i][0], mmap_mode='r')
            temp_label = np.load(path_name[t_idx][i][1], mmap_mode='r')
            test_data.append(temp_data)
            test_label.append(temp_label)
    test_data = np.concatenate(test_data, axis=0)
    test_label = np.concatenate(test_label, axis=0)
    assert test_data.shape[0] == test_label.shape[0], "test data and test size do not match!"
    print("Number of test samples:", test_data.shape)

    return train_data, train_label, finetune_train_data, finetune_train_label, finetune_val_data, finetune_val_label, test_data, test_label  #should return a list of samples here
