import numpy as np
import os

# =========================
# Paths (server)
# =========================
BASE_DIR = 'data'
D3_DIR = os.path.join(BASE_DIR, 'd3')              # input of step4
WINDOW_DIR = os.path.join(BASE_DIR, 'window')      # output of step4 / input of step5
MODEL_INPUT_DIR = os.path.join(BASE_DIR, 'model_input')  # output of step5

os.makedirs(WINDOW_DIR, exist_ok=True)
os.makedirs(MODEL_INPUT_DIR, exist_ok=True)

# =========================
# Step4: build window segments
# (original code, just un-commented + paths changed)
# =========================
win_step = 1
win_length = 3

eeg_segments_number = 10
fnirs_segments_number = 22

eeg_srate = 200
fnirs_srate = 10

subject_list = sorted([f for f in os.listdir(D3_DIR) if f.endswith('.npz')])

for subject in subject_list:
    with np.load(os.path.join(D3_DIR, subject)) as data:
        eeg = data['eeg']
        hbo = data['hbo']
        hbr = data['hbr']
        label = data['label']

    eeg_window = np.ones((60, eeg_segments_number, 16, 16, win_length * eeg_srate))
    hbo_window = np.ones((60, fnirs_segments_number, 16, 16, win_length * fnirs_srate))
    hbr_window = np.ones((60, fnirs_segments_number, 16, 16, win_length * fnirs_srate))

    for e in range(60):
        # EEG: 10 windows
        for w in range(eeg_segments_number):
            eeg_start_indice = (3 + w) * eeg_srate
            eeg_end_indice = eeg_start_indice + win_length * eeg_srate
            eeg_segment = eeg[e, :, :, eeg_start_indice:eeg_end_indice]
            eeg_window[e, w, :, :, :] = eeg_segment

        # fNIRS: 22 windows
        for fw in range(fnirs_segments_number):
            fnirs_start_indice = (3 + fw) * fnirs_srate
            fnirs_end_indice = fnirs_start_indice + win_length * fnirs_srate
            hbo_segment = hbo[e, :, :, fnirs_start_indice:fnirs_end_indice]
            hbr_segment = hbr[e, :, :, fnirs_start_indice:fnirs_end_indice]
            hbo_window[e, fw, :, :, :] = hbo_segment
            hbr_window[e, fw, :, :, :] = hbr_segment

    print("STEP4 window shapes:", eeg_window.shape, hbo_window.shape, hbr_window.shape, label.shape)

    save_dict = {
        'eeg': eeg_window,
        'hbo': hbo_window,
        'hbr': hbr_window,
        'label': label
    }

    np.savez(os.path.join(WINDOW_DIR, subject), **save_dict)
    print(f'\n==============save window {subject} success=============\n')

# =========================
# Step5: build model_input (EGTA lag)
# (original code, just paths changed)
# =========================
fnirs_lag_length = 11  # with t-self

subject_list = sorted([f for f in os.listdir(WINDOW_DIR) if f.endswith('.npz')])

for subject in subject_list:
    with np.load(os.path.join(WINDOW_DIR, subject)) as data:
        eeg = data['eeg']
        hbo = data['hbo']
        hbr = data['hbr']
        label = data['label']

    # eeg
    eeg_session_dataset = np.expand_dims(eeg, axis=-1)
    eeg_input = eeg_session_dataset.reshape(600, 16, 16, 600, 1)

    # fnirs
    fnirs_session_dataset = np.ones((60, 10, fnirs_lag_length, 16, 16, 30, 2))

    for e in range(60):
        for w in range(10):
            # first 10 windows has same time interval
            hbo_sample = hbo[e, w:(w + fnirs_lag_length),]
            hbr_sample = hbr[e, w:(w + fnirs_lag_length),]

            hbo_sample = np.expand_dims(hbo_sample, axis=-1)
            hbr_sample = np.expand_dims(hbr_sample, axis=-1)

            fnirs_sample = np.concatenate((hbo_sample, hbr_sample), axis=-1)
            fnirs_session_dataset[e, w,] = fnirs_sample

    fnirs_input = fnirs_session_dataset.reshape(600, 11, 16, 16, 30, 2)

    # label
    label_session_dataset = label.T
    label_input = np.repeat(label_session_dataset, repeats=10, axis=0)

    print("STEP5 model_input shapes:", eeg_input.shape, fnirs_input.shape, label_input.shape)

    save_dict = {
        'eeg': eeg_input,
        'fnirs': fnirs_input,
        'label': label_input
    }

    np.savez(os.path.join(MODEL_INPUT_DIR, subject), **save_dict)
    print(f'\n==============save model_input {subject} success=============\n')

