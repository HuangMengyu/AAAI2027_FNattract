from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


dir_path = f'SleepEDFx/SleepCassette'
f_names = os.listdir(dir_path)

psg_f_names = []
label_f_names = []
for f_name in f_names:
    if 'PSG' in f_name:
        psg_f_names.append(f_name)
    if 'Hypnogram' in f_name:
        label_f_names.append(f_name)

psg_f_names.sort()
label_f_names.sort()

psg_label_f_pairs = []
for psg_f_name, label_f_name in zip(psg_f_names, label_f_names):
    if psg_f_name[:6] == label_f_name[:6]:
        psg_label_f_pairs.append((psg_f_name, label_f_name))

label2id = {'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,
            'Sleep stage R': 4}

save_dir = f"SleepEDFx/SleepCassette_processed"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

SEQ_LENGTH = 1
total_num_of_seq = 0
signal_name = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']
for psg_f_name, label_f_name in tqdm(psg_label_f_pairs):
    epochs_list = []
    labels_list = []

    raw = read_raw_edf(os.path.join(dir_path, psg_f_name), preload=True)
    raw.pick_channels(signal_name)
    raw.filter(0.3, 35, fir_design='firwin')
    annotation = mne.read_annotations(os.path.join(dir_path, label_f_name))
    raw.set_annotations(annotation, emit_warning=False)

    events_train, event_id = mne.events_from_annotations(
        raw, chunk_duration=30.)
    if 'Sleep stage ?' in event_id.keys():
        event_id.pop('Sleep stage ?')
    if 'Movement time' in event_id.keys():
        event_id.pop('Movement time')
    tmax = 30. - 1. / raw.info['sfreq']  # tmax in included
    epochs_train = mne.Epochs(raw=raw, events=events_train,
                              event_id=event_id, tmin=0., tmax=tmax, baseline=None)
    print(epochs_train.event_id)
    labels = []
    for epoch_annotation in epochs_train.get_annotations_per_epoch():
        labels.append(epoch_annotation[0][2])
    length = len(labels)

    a = []
    for i, label in enumerate(labels):
        if label != 'Sleep stage W':
            a.append(i)
    print(len(a))
    print(0, a[0], a[-1], length)
    if a[0] - 60 >= 0:
        start = a[0] - 60
    else:
        start = 0

    if a[-1] + 60 < length:
        end = a[-1] + 60
    else:
        end = length
    print(start, end)

    epochs = epochs_train[start:end]
    labels_ = labels[start:end]

    for epoch in epochs:
        epochs_list.append(epoch)
    for label in labels_:
        labels_list.append(label2id[label])

    epochs_seq = np.array(epochs_list)
    labels_seq = np.array(labels_list)
    print(epochs_seq.shape, labels_seq.shape)


    sub_save_dir = save_dir + f"/{psg_f_name[3:5]}" + "/data"
    sub_save_dir_label = save_dir + f"/{psg_f_name[3:5]}" + "/label"

    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir, exist_ok=True)

    if not os.path.exists(sub_save_dir_label):
        os.makedirs(sub_save_dir_label, exist_ok=True)
    
    save_data_dir = sub_save_dir + f"/{psg_f_name[5]}.npy" # one npy file for one subject
    save_label_dir = sub_save_dir_label + f"/{psg_f_name[5]}.npy"
    np.save(save_data_dir, epochs_seq)
    np.save(save_label_dir, labels_seq)

    total_num_of_seq += labels_seq.shape[0]
    print(f"Current subject: {psg_f_name[3:5]}, Total number of sequences: {total_num_of_seq}")

print("Finished!")
