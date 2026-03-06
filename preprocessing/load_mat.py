import scipy.io as io
import numpy as np
import os

subject_list = []

folder_path = r'data/olddataset/EEG_01-26'
for filename in os.listdir(folder_path):
    subject_no = filename.split('-')[0]

    subject_list.append(subject_no)

for name in subject_list:
    eeg_data = io.loadmat(r'data/olddataset/EEG_01-26/{}-EEG/cnt_wg.mat'.format(name))
    eeg_mrk_data = io.loadmat(r'data/olddataset/EEG_01-26/{}-EEG/mrk_wg.mat'.format(name))

    fnirs_data = io.loadmat(r'data/olddataset/NIRS_01-26/{}-NIRS/cnt_wg.mat'.format(name))
    fnirs_mrk_data = io.loadmat(r'data/olddataset/NIRS_01-26/{}-NIRS/mrk_wg.mat'.format(name))

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

    save_dir = r'data/mat2array'
    save_name = name

    np.savez(os.path.join(save_dir,save_name),**save_dict)
    print('\n==============save {} success=============\n'.format(save_name))







