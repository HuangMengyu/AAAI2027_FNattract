# encoding=utf-8

import os
import numpy as np
import pandas as pd
import torch
import pickle as cp
from torch.utils.data import Dataset, DataLoader
from utils import get_sample_weights, opp_sliding_window
from scipy.signal import medfilt, butter, filtfilt


NUM_FEATURES = 52

class data_loader_pamap2(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return sample, target

    def __len__(self):
        return len(self.samples)


def normalize(x):
    """Normalizes all sensor channels by mean substraction,
    dividing by the standard deviation and by 2.
    Z score normalization

    :param x: numpy integer matrix
        Sensor data
    :return:
        Normalized sensor data
    """

    T, C = x.shape

    for c in range(C):
        x[:, c] = np.array(x[:, c], dtype=np.float32)
        m = np.mean(x[:, c], axis=0)
        x[:, c] -= m
        std = np.std(x[:, c], axis=0)
        std += 0.000001

        x[:, c] /= std

    return x


def complete_HR(data):
    """Sampling rate for the heart rate is different from the other sensors. Missing
    measurements are filled

    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        HR channel data
    """

    pos_NaN = np.isnan(data)
    idx_NaN = np.where(pos_NaN == False)[0]
    data_no_NaN = data * 0
    for idx in range(idx_NaN.shape[0] - 1):
        data_no_NaN[idx_NaN[idx]: idx_NaN[idx + 1]] = data[idx_NaN[idx]]

    data_no_NaN[idx_NaN[-1]:] = data[idx_NaN[-1]]

    return data_no_NaN


def divide_x_y(data):
    """Segments each sample into time, labels and sensor channels

    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        Time and labels as arrays, sensor channels as matrix
    """
    data_t = data[:, 0]
    data_y = data[:, 1]
    data_x = data[:, 2:]

    return data_t, data_x, data_y

def adjust_idx_labels(data_y):
    """The pamap2 dataset contains in total 24 action classes. However, for the protocol,
    one uses only 9 action classes. This function adjust the labels picking the labels
    for the protocol settings

    :param data_y: numpy integer array
        Sensor labels
    :return: numpy integer array
        Modified sensor labels
    """

    data_y[data_y == 24] = 0
    data_y[data_y == 12] = 1
    data_y[data_y == 13] = 8


    return data_y


def del_labels(data_t, data_x, data_y):
    """The pamap2 dataset contains in total 24 action classes. However, for the protocol,
    one uses only 9 action classes. This function deletes the nonrelevant labels

    18 -> 9

    :param data_y: numpy integer array
        Sensor labels
    :return: numpy integer array
        Modified sensor labels
    """

    idy = np.where(data_y == 0)[0]
    labels_delete = idy

    idy = np.where(data_y == 8)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 9)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 10)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 11)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 18)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 19)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 20)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    # add the following lines to output only 9 classes
    idy = np.where(data_y == 1)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 16)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 17)[0]
    labels_delete = np.concatenate([labels_delete, idy])


    return np.delete(data_t, labels_delete, 0), np.delete(data_x, labels_delete, 0), np.delete(data_y, labels_delete, 0)


def downsampling(data_t, data_x, data_y):
    """Recordings are downsamplied to 30Hz, as in the Opportunity dataset

    :param data_t: numpy integer array
        time array
    :param data_x: numpy integer array
        sensor recordings
    :param data_y: numpy integer array
        labels
    :return: numpy integer array
        Downsampled input
    """

    idx = np.arange(0, data_t.shape[0], 3)

    return data_t[idx], data_x[idx], data_y[idx]



def interpolate_nans(X, max_gap=100):
    """
    Linear interpolation per channel.
    
    Args:
        X: np.ndarray of shape (T, C)
        max_gap: maximum allowed consecutive NaNs (in samples).
                 If exceeded, the segment should be removed later.
                 Use None to interpolate all gaps.
    Returns:
        X_interp: interpolated array
        valid_mask: boolean mask of valid samples (True = keep)
    """
    T, C = X.shape
    X_interp = X.copy()
    valid_mask = np.ones(T, dtype=bool)

    for c in range(C):
        s = pd.Series(X[:, c])

        nan_groups = s.isna().astype(int).groupby(
            s.notna().astype(int).cumsum()
        ).sum()

        if max_gap is not None:
            long_gaps = nan_groups[nan_groups > max_gap].index
            for g in long_gaps:
                idx = s.index[s.notna().astype(int).cumsum() == g]
                valid_mask[idx] = False

        X_interp[:, c] = s.interpolate(
            method="linear",
            limit_direction="both"
        ).to_numpy()

    return X_interp, valid_mask




def process_dataset_file(data):
    """Function defined as a pipeline to process individual Pamap2 files

    :param data: numpy integer matrix
        channel data: samples in rows and sensor channels in columns
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into samples-channel measurements (x) and labels (y)
    """

    # Data is divided in time, sensor data and labels
    data_t, data_x, data_y = divide_x_y(data)

    # nonrelevant labels are deleted
    data_t, data_x, data_y = del_labels(data_t, data_x, data_y)

    # Labels are adjusted
    data_y = adjust_idx_labels(data_y)
    data_y = data_y.astype(int)

    if data_x.shape[0] != 0:
        # extract only the relevant channels
        hand_acce = data_x[:, 2:5]
        hand_gyro = data_x[:, 8:11]
        chest_acce = data_x[:, 19:22]
        chest_gyro = data_x[:, 25:28]
        ankle_acce = data_x[:, 36:39]
        ankle_gyro = data_x[:, 42:45]

        data_multimodal_x = np.concatenate([hand_acce, chest_acce, ankle_acce, hand_gyro, chest_gyro, ankle_gyro], axis=1)
        print("data_multimodal_x shape before NaN removal: {}".format(data_multimodal_x.shape))
        
        # interpolate NaN values instead of dropping them
        data_multimodal_x, mask = interpolate_nans(data_multimodal_x, max_gap=100)
        data_multimodal_x = data_multimodal_x[mask]
        data_y = data_y[mask]
        data_t = data_t[mask]
        print("data_multimodal_x shape after NaN removal: {}".format(data_multimodal_x.shape))
        
        data_x = normalize(data_multimodal_x)

    else:
        data_x = data_x
        data_y = data_y
        data_t = data_t

        print("SIZE OF THE SEQUENCE IS ZERO")

    return data_x, data_y


def load_data_pamap2():
    dataset = 'PAMAP2_Dataset/'
    # File names of the files defining the PAMAP2 data.
    PAMAP2_DATA_FILES = ['PAMAP2_Dataset/Protocol/subject101.dat',  # 0
                            'PAMAP2_Dataset/Optional/subject101.dat',  # 1
                            'PAMAP2_Dataset/Protocol/subject102.dat',  # 2
                            'PAMAP2_Dataset/Protocol/subject103.dat',  # 3
                            'PAMAP2_Dataset/Protocol/subject104.dat',  # 4
                            'PAMAP2_Dataset/Protocol/subject107.dat',  # 5
                            'PAMAP2_Dataset/Protocol/subject108.dat',  # 6
                            'PAMAP2_Dataset/Optional/subject108.dat',  # 7
                            'PAMAP2_Dataset/Protocol/subject109.dat',  # 8
                            'PAMAP2_Dataset/Optional/subject109.dat',  # 9
                            'PAMAP2_Dataset/Protocol/subject105.dat',  # 10
                            'PAMAP2_Dataset/Optional/subject105.dat',  # 11
                            'PAMAP2_Dataset/Protocol/subject106.dat',  # 12
                            'PAMAP2_Dataset/Optional/subject106.dat',  # 13
                            ]

        

    print('Processing dataset files ...')

    data_dict = dict()
    for filename in PAMAP2_DATA_FILES:
        subject_id = filename.split('subject')[1].split('.dat')[0]
        print('Processing subject: {}'.format(subject_id))
        if subject_id not in data_dict.keys():
            data_dict[subject_id] = {'data': [], 'labels': []}

        data = np.loadtxt(dataset + filename)
        x, y = process_dataset_file(data)
        print(x.shape)
        print(y.shape)
        if x.shape[0] != 0:
            data_dict[subject_id]['data'].append(x) 
            data_dict[subject_id]['labels'].append(y)

    print(data_dict.keys())
    data_dict_window = dict()

    for subject_id in data_dict.keys():
        data_dict[subject_id]['data'] = np.concatenate(data_dict[subject_id]['data'], axis=0)
        data_dict[subject_id]['labels'] = np.concatenate(data_dict[subject_id]['labels'], axis=0)
        
        data_dict_window[subject_id] = {'data': [], 'labels': []}
        data_dict_window[subject_id]['data'], data_dict_window[subject_id]['labels'] = opp_sliding_window(data_dict[subject_id]['data'],
                                                          data_dict[subject_id]['labels'],
                                                           ws = 256, ss = 128)
        N,T,F = data_dict_window[subject_id]['data'].shape
        data_dict_window[subject_id]['data'] = data_dict_window[subject_id]['data'].transpose(0, 2, 1)
        
        print('Subject {}: data shape {}, labels shape {}'.format(subject_id,
                                                                 data_dict_window[subject_id]['data'].shape,
                                                                 data_dict_window[subject_id]['labels'].shape))

        ## Save everything
        output_dir = 'PAMAP2_processed/'
        sub_output_dir = os.path.join(output_dir, f'{subject_id}')
        if not os.path.exists(sub_output_dir):
            os.makedirs(sub_output_dir, exist_ok=True)
        
        data_sub_output_dir = os.path.join(sub_output_dir, 'data')
        label_sub_output_dir = os.path.join(sub_output_dir, 'label')
        if not os.path.exists(data_sub_output_dir):
            os.makedirs(data_sub_output_dir, exist_ok=True)
        if not os.path.exists(label_sub_output_dir):
            os.makedirs(label_sub_output_dir, exist_ok=True)
        
        print("Data shape:", data_dict_window[subject_id]['data'].shape, "Label shape:", data_dict_window[subject_id]['labels'].shape)
        
        np.save(os.path.join(data_sub_output_dir, f'0.npy'), data_dict_window[subject_id]['data'])
        np.save(os.path.join(label_sub_output_dir, f'0.npy'), data_dict_window[subject_id]['labels'])
    
    return 



def load_fixed_slidwin_balancedUp(dataset="pamap2", batch_size=64, SLIDING_WINDOW_LEN=30, SLIDING_WINDOW_STEP=15):

    if dataset == "pamap2":
        x_train, y_train, x_val, y_val, x_test, y_test = load_data_pamap2()  # (557963, 113), (557963,), (118750, 113), (118750, )

        x_train_win, y_train_win = opp_sliding_window(x_train, y_train, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        x_val_win, y_val_win = opp_sliding_window(x_val, y_val, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
        x_test_win, y_test_win = opp_sliding_window(x_test, y_test, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)

        unique_ytrain, counts_ytrain = np.unique(y_train_win, return_counts=True)

        weights = 100.0/ torch.Tensor(counts_ytrain)
        weights = weights.double()
        sample_weights = get_sample_weights(y_train_win, weights)

        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_set = data_loader_pamap2(x_train_win, y_train_win)
        val_set = data_loader_pamap2(x_val_win, y_val_win)
        test_set = data_loader_pamap2(x_test_win, y_test_win)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        print('train_loader batch: ', len(train_loader), 'test_loader batch: ', len(test_loader))
        return train_loader, val_loader, test_loader

if __name__ == '__main__':
    torch.manual_seed(10)

    load_data_pamap2()
