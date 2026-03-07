import numpy as np
import os
import mne
import matplotlib.pyplot as plt  # 保留导入以尽量贴近原脚本（但不再使用 show）

# =========================
# Paths (server)
# =========================
BASE_DIR = 'data'
IN_DIR = os.path.join(BASE_DIR, 'mat2array')
OUT_DIR = os.path.join(BASE_DIR, 'preprocessed')
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# EEG info
# =========================
eeg_chn_names = [
    'Fp1','AFF5h','AFz','F1','FC5','FC1','T7','C3','Cz','CP5','CP1','P7','P3','Pz','POz','O1','Fp2',
    'AFF6h','F2','FC2','FC6','C4','T8','CP2','CP6','P4','P8','O2'
]
eeg_info = mne.create_info(ch_names=eeg_chn_names, sfreq=200, ch_types='eeg')
eeg_info.set_montage('standard_1005')

# =========================
# fNIRS info
# =========================
fnirs_chn_names = [
    'AF7','AFF5','AFp7','AF5h','AFp3','AFF3h','AF1','AFFz','AFpz','AF2','AFp4',
    'FCC3','C3h','C5h','CCP3','CPP3','P3h','P5h','PPO3',
    'AFF4h','AF6h','AFF6','AFp8','AF8',
    'FCC4','C6h','C4h','CCP4','CPP4','P6h','P4h','PPO4',
    'PPOz','PO1','PO2','POOz'
]
fnirs_info = mne.create_info(ch_names=fnirs_chn_names, sfreq=10, ch_types='eeg')
fnirs_info.set_montage('standard_1005')

# =========================
# Subjects
# =========================
subject_list = sorted([f[:-4] for f in os.listdir(IN_DIR) if f.endswith('.npz')])
subject_list.sort()
print(f'Found {len(subject_list)} subjects in {IN_DIR}')

for subject_no in subject_list:
    in_path = os.path.join(IN_DIR, f'{subject_no}.npz')
    with np.load(in_path) as data:
        eeg = data['eeg']
        eeg_time = data['eeg_time']
        hbo = data['hbo']
        hbr = data['hbr']
        fnirs_time = data['fnirs_time']
        label = data['label']

    # =========================
    # EEG
    # =========================
    raw = mne.io.RawArray(data=eeg[:-2, :], info=eeg_info)

    raw_notch = raw.notch_filter(np.arange(50, 100, 50))
    raw_filtered = raw_notch.filter(
        0.5, 50., method='iir',
        iir_params=dict(order=6, ftype='butter')
    )

    raw_avg_ref = raw_filtered.set_eeg_reference(ref_channels="average")
    raw_avg_ref.load_data()

    # filtering just for ICA (保留原逻辑)
    filt_ica_raw = raw_avg_ref.copy().filter(l_freq=1., h_freq=None)

    ica = mne.preprocessing.ICA(n_components=20)
    ica.fit(filt_ica_raw)

    ica.plot_components()
    ica.plot_sources(raw_avg_ref)
    ica.plot_properties(raw_avg_ref)
    plt.show()

    input_str = input('exclude components:')
    exclude_list = input_str.split(" ")
    for j in range(0, len(exclude_list)):
        exclude_list[j] = int(exclude_list[j])

    ica.exclude = exclude_list
    print(ica.exclude)
    raw_icaed = ica.apply(raw_avg_ref)

    eeg_processed = raw_icaed.get_data()

    # =========================
    # fNIRS
    # =========================
    hbo_raw = mne.io.RawArray(data=hbo, info=fnirs_info)
    hbr_raw = mne.io.RawArray(data=hbr, info=fnirs_info)

    hbo_filtered = hbo_raw.filter(
        0.01, 0.1, method='iir',
        iir_params=dict(order=6, ftype='butter')
    )
    hbr_filtered = hbr_raw.filter(
        0.01, 0.1, method='iir',
        iir_params=dict(order=6, ftype='butter')
    )

    hbo_processed = hbo_filtered.get_data()
    hbr_processed = hbr_filtered.get_data()

    save_dict = {
        'eeg': eeg_processed,
        'eeg_time': eeg_time,
        'hbo': hbo_processed,
        'hbr': hbr_processed,
        'fnirs_time': fnirs_time,
        'label': label
    }

    out_path = os.path.join(OUT_DIR, f'{subject_no}.npz')
    np.savez(out_path, **save_dict)
    print(f'\n==============save {subject_no} success=============\n')
