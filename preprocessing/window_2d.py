from joblib import Parallel, delayed
import numpy as np
import os

# =========================
# Paths
# =========================
BASE_DIR = 'data'
EPOCH_DIR = os.path.join(BASE_DIR, 'epoch')          # now we read epoch directly
WINDOW_DIR = os.path.join(BASE_DIR, 'window_2d')
MODEL_INPUT_DIR = os.path.join(BASE_DIR, 'model_input_2d')

os.makedirs(WINDOW_DIR, exist_ok=True)
os.makedirs(MODEL_INPUT_DIR, exist_ok=True)

# =========================
# Parameters
# =========================
win_length = 3

eeg_segments_number = 10
fnirs_segments_number = 22

eeg_srate = 200
fnirs_srate = 10

n_epoch = 60

eeg_channels = 28
fnirs_channels = 36

subject_list = sorted([f for f in os.listdir(EPOCH_DIR) if f.endswith('.npz')])


# ==========================================================
# STEP4: build sliding windows (no 16x16 spatial interpolation)
# ==========================================================
def process1(subject):

    with np.load(os.path.join(EPOCH_DIR, subject)) as data:
        eeg = data['eeg']   # (60, 28, time)
        hbo = data['hbo']   # (60, 36, time)
        hbr = data['hbr']   # (60, 36, time)
        label = data['label']

    eeg_window = np.ones((n_epoch, eeg_segments_number, eeg_channels, win_length * eeg_srate))
    hbo_window = np.ones((n_epoch, fnirs_segments_number, fnirs_channels, win_length * fnirs_srate))
    hbr_window = np.ones((n_epoch, fnirs_segments_number, fnirs_channels, win_length * fnirs_srate))

    for e in range(n_epoch):

        # EEG windows
        for w in range(eeg_segments_number):

            start = (3 + w) * eeg_srate
            end = start + win_length * eeg_srate

            eeg_segment = eeg[e, :, start:end]

            eeg_window[e, w] = eeg_segment

        # fNIRS windows
        for w in range(fnirs_segments_number):

            start = (3 + w) * fnirs_srate
            end = start + win_length * fnirs_srate

            hbo_segment = hbo[e, :, start:end]
            hbr_segment = hbr[e, :, start:end]

            hbo_window[e, w] = hbo_segment
            hbr_window[e, w] = hbr_segment

    print("STEP4 window shapes:", eeg_window.shape, hbo_window.shape, hbr_window.shape)

    save_dict = {
        'eeg': eeg_window,
        'hbo': hbo_window,
        'hbr': hbr_window,
        'label': label
    }

    np.savez(os.path.join(WINDOW_DIR, subject), **save_dict)

    print(f'\n============== save window {subject} success ==============\n')


Parallel(n_jobs=32)(
    delayed(process1)(subject)
    for subject in subject_list
)


# ==========================================================
# STEP5: build model_input (EGTA lag)
# ==========================================================

fnirs_lag_length = 11

subject_list = sorted([f for f in os.listdir(WINDOW_DIR) if f.endswith('.npz')])


def process2(subject):

    with np.load(os.path.join(WINDOW_DIR, subject)) as data:
        eeg = data['eeg']   # (60,10,28,600)
        hbo = data['hbo']   # (60,22,36,30)
        hbr = data['hbr']
        label = data['label']

    # =========================
    # EEG input
    # =========================

    eeg_session_dataset = np.expand_dims(eeg, axis=-1)

    # (60,10,28,600,1) -> (600,28,600,1)
    eeg_input = eeg_session_dataset.reshape(600, eeg_channels, 600, 1)


    # =========================
    # fNIRS input
    # =========================

    fnirs_session_dataset = np.ones((60, 10, fnirs_lag_length, fnirs_channels, 30, 2))

    for e in range(60):
        for w in range(10):

            hbo_sample = hbo[e, w:(w + fnirs_lag_length)]
            hbr_sample = hbr[e, w:(w + fnirs_lag_length)]

            hbo_sample = np.expand_dims(hbo_sample, axis=-1)
            hbr_sample = np.expand_dims(hbr_sample, axis=-1)

            fnirs_sample = np.concatenate((hbo_sample, hbr_sample), axis=-1)

            fnirs_session_dataset[e, w] = fnirs_sample

    # (60,10,11,36,30,2) -> (600,11,36,30,2)
    fnirs_input = fnirs_session_dataset.reshape(600, fnirs_lag_length, fnirs_channels, 30, 2)


    # =========================
    # label
    # =========================

    label_session_dataset = label.T
    label_input = np.repeat(label_session_dataset, repeats=10, axis=0)

    print("STEP5 model_input shapes:", eeg_input.shape, fnirs_input.shape, label_input.shape)

    save_dict = {
        'eeg': eeg_input,
        'fnirs': fnirs_input,
        'label': label_input
    }

    np.savez(os.path.join(MODEL_INPUT_DIR, subject), **save_dict)

    print(f'\n============== save model_input {subject} success ==============\n')


Parallel(n_jobs=32)(
    delayed(process2)(subject)
    for subject in subject_list
)