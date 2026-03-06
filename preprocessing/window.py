import numpy as np
import os

'''
win_step = 1
win_length = 3

eeg_segments_number = 10
fnirs_segments_number = 22

eeg_srate = 200
fnirs_srate = 10

subject_path = r'data/d3'
subject_list = os.listdir(subject_path)

for subject in subject_list:
    with np.load(os.path.join(subject_path, subject)) as data:
        eeg = data['eeg']
        hbo = data['hbo']
        hbr = data['hbr']
        label = data['label']

    eeg_window = np.ones((60, eeg_segments_number, 16, 16, win_length*eeg_srate))
    hbo_window = np.ones((60, fnirs_segments_number, 16, 16, win_length*fnirs_srate))
    hbr_window = np.ones((60, fnirs_segments_number, 16, 16, win_length*fnirs_srate))

    for e in range(60):
        # first 10 windows has same time interval
        for w in range(eeg_segments_number):
            eeg_start_indice = (3+w)*eeg_srate
            eeg_end_indice = eeg_start_indice + win_length*eeg_srate

            eeg_segment = eeg[e, :, :, eeg_start_indice:eeg_end_indice]

            eeg_window[e, w, :, :, :] = eeg_segment

        for fw in range(fnirs_segments_number):
            fnirs_start_indice = (3+fw)*fnirs_srate
            fnirs_end_indice = fnirs_start_indice + win_length*fnirs_srate

            hbo_segment = hbo[e, :, :, fnirs_start_indice:fnirs_end_indice]
            hbr_segment = hbr[e, :, :, fnirs_start_indice:fnirs_end_indice]

            hbo_window[e, fw, :, :, :] = hbo_segment
            hbr_window[e, fw, :, :, :] = hbr_segment

    print(eeg_window.shape)
    print(hbo_window.shape)
    print(hbr_window.shape)
    print(label.shape)
    
    save_dict = {
    'eeg':eeg_window,
    'hbo':hbo_window,
    'hbr':hbr_window,
    'label':label
    }

    save_dir = r'data/window'
    save_name = subject

    np.savez(os.path.join(save_dir,save_name),**save_dict)
    print('\n==============save {} success=============\n'.format(save_name)) 
'''



fnirs_lag_length = 11 # with t-self

subject_path = r'data/window'
subject_list = os.listdir(subject_path)

for subject in subject_list:
    with np.load(os.path.join(subject_path, subject)) as data:
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
            hbo_sample = hbo[e, w:(w+fnirs_lag_length),]
            hbr_sample = hbr[e, w:(w+fnirs_lag_length),]

            hbo_sample = np.expand_dims(hbo_sample, axis=-1)
            hbr_sample = np.expand_dims(hbr_sample, axis=-1)

            fnirs_sample = np.concatenate((hbo_sample, hbr_sample), axis=-1)

            fnirs_session_dataset[e, w,] = fnirs_sample
    
    fnirs_input = fnirs_session_dataset.reshape(600, 11, 16, 16, 30, 2)

    # label
    label_session_dataset = label.T
    label_input = np.repeat(label_session_dataset, repeats=10, axis=0)

    print(eeg_input.shape)
    print(fnirs_input.shape)
    print(label_input.shape)

    save_dict = {
    'eeg':eeg_input,
    'fnirs':fnirs_input,
    'label':label_input
    }

    save_dir = r'data/model_input'
    save_name = subject

    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir,save_name),**save_dict)
    print('\n==============save {} success=============\n'.format(save_name)) 


    


    
            




    