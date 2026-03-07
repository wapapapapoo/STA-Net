import scipy.io as io
import numpy as np
import os

subject_list = []

# ===== base paths on server =====
BASE_DIR = '/home/libiao/sta/olddataset'
EEG_DIR = os.path.join(BASE_DIR, 'EEG_01-26')
NIRS_DIR = os.path.join(BASE_DIR, 'NIRS_01-26')

# output directory for step0 (mat -> npz)
OUT_DIR = '/home/libiao/sta/STA-Net-work/data/step0_mat2array'
os.makedirs(OUT_DIR, exist_ok=True)

# list subjects from EEG directory
folder_path = EEG_DIR

for filename in os.listdir(folder_path):
    subject_no = filename.split('-')[0]

    subject_list.append(subject_no)

for name in subject_list:
    eeg_data = io.loadmat(os.path.join(EEG_DIR, f'{name}-EEG', 'cnt_wg.mat'))
    eeg_mrk_data = io.loadmat(os.path.join(EEG_DIR, f'{name}-EEG', 'mrk_wg.mat'))

    fnirs_data = io.loadmat(os.path.join(NIRS_DIR, f'{name}-NIRS', 'cnt_wg.mat'))
    fnirs_mrk_data = io.loadmat(os.path.join(NIRS_DIR, f'{name}-NIRS', 'mrk_wg.mat'))

    eeg = eeg_data['cnt_wg'][0,0][3].T
    eeg_time = eeg_mrk_data['mrk_wg'][0,0][0]

    hbo = fnirs_data['cnt_wg']['oxy'][0,0][0,0][5].T
    hbr = fnirs_data['cnt_wg']['deoxy'][0,0][0,0][5].T
    fnirs_time = fnirs_mrk_data['mrk_wg'][0,0][0]

    label = eeg_mrk_data['mrk_wg'][0,0][1]

    print(eeg.shape)
    print(eeg_time.shape)

    print(hbo.shape)
    print(hbr.shape)
    print(fnirs_time.shape)

    print(label.shape)

    save_dict = {
        'eeg':eeg,
        'eeg_time':eeg_time,
        'hbo':hbo,
        'hbr':hbr,
        'fnirs_time':fnirs_time,
        'label':label
    }

    save_dir = OUT_DIR
    save_name = name

    np.savez(os.path.join(save_dir, save_name), **save_dict)

    print('\n==============save {} success!=============\n'.format(save_name))







