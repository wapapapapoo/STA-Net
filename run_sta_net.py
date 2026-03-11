from sta import sta_net

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
import os

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class targetacccallback(keras.callbacks.Callback):
    def __init__(self, target_acc):
        super().__init__()

        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs={}):
        if(logs['class_output_loss'] <= self.target_acc):
            # print("\nReached target loss value {} so cancelling training!\n".format(self.target_acc))
            self.model.stop_training = True
class PlateauAveraging(tf.keras.callbacks.Callback):

    def __init__(
        self,
        monitor="val_class_output_loss",
        window=9,
        offset=5,
        min_delta=0,
        patience=20,
        trim_ratio=0.2
    ):
        super().__init__()

        self.monitor = monitor
        self.window = window
        self.offset = offset
        self.min_delta = min_delta
        self.patience = patience
        self.trim_ratio = trim_ratio

        self.loss_history = []
        self.weight_history = []

        self.plateau_started = False
        self.wait = 0

    def trimmed_mean(self, arr):
        arr = np.sort(arr)
        k = int(len(arr) * self.trim_ratio)
        if k > 0:
            arr = arr[:-k]
        return np.mean(arr)

    def moving_avg(self):
        if len(self.loss_history) < self.window:
            return None
        window_vals = self.loss_history[-self.window:]
        return self.trimmed_mean(window_vals)

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs[self.monitor]
        self.loss_history.append(val_loss)
        self.weight_history.append(self.model.get_weights())
        smooth = self.moving_avg()
        if smooth is None:
            return
        if len(self.loss_history) > self.window:
            prev_window = self.loss_history[-self.window-1:-1]
            prev = self.trimmed_mean(prev_window)
            improvement = prev - smooth
            if improvement < self.min_delta:
                if not self.plateau_started:
                    print(f"; Plateau detected at epoch {epoch+1}")
                    self.plateau_started = True
        if self.plateau_started:
            self.wait += 1
            if self.wait >= self.patience:
                print("; Plateau stable → stopping training")
                self.stop_epoch = epoch
                self.model.stop_training = True

    def average_weights(self, weights_list):
        avg = [w.copy() for w in weights_list[0]]
        for weights in weights_list[1:]:
            for i in range(len(avg)):
                avg[i] += weights[i]
        for i in range(len(avg)):
            avg[i] /= len(weights_list)
        return avg

    def on_train_end(self, logs=None):
        stop = self.stop_epoch + 1
        start = stop - self.offset - self.window
        end = stop - self.offset
        start = max(start, 0)
        selected = self.weight_history[start:end]
        if len(selected) == 0:
            print("; Not enough models for averaging")
            return
        print(f"; Averaging epochs {start+1} → {end}")
        avg_weights = self.average_weights(selected)
        self.model.set_weights(avg_weights)


class TrainPlateauSWA(tf.keras.callbacks.Callback):

    def __init__(
        self,
        monitor="class_output_accuracy",
        patience=15,   # n
        swa_k=8,       # k
        offset=2       # 跳过最后offset个epoch
    ):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.swa_k = swa_k
        self.offset = offset

        self.best = -np.inf
        self.wait = 0

        self.weight_buffer = []

    def on_epoch_end(self, epoch, logs=None):
        acc = logs[self.monitor]

        # 保存权重
        weights = self.model.get_weights()
        self.weight_buffer.append([w.copy() for w in weights])

        # buffer长度需要 ≥ k + offset
        max_buffer = self.swa_k + self.offset
        if len(self.weight_buffer) > max_buffer:
            self.weight_buffer.pop(0)

        # 判断提升
        if acc > self.best:
            self.best = acc
            self.wait = 0
        else:
            self.wait += 1

        # plateau
        if self.wait >= self.patience:
            print("; Train acc plateau → stopping")
            buf = self.weight_buffer
            if len(buf) <= self.offset:
                # 不够offset，只能全平均
                selected = buf
            else:
                end = len(buf) - self.offset
                start = max(0, end - self.swa_k)
                selected = buf[start:end]
            k = len(selected)
            avg_weights = []
            for ws in zip(*selected):
                avg_weights.append(np.mean(ws, axis=0))
            print(f"; SWA averaging {k} epochs (offset={self.offset})")
            self.model.set_weights(avg_weights)
            self.model.stop_training = True


def sample_segments(total_len, segment_len=25, num_segments=4):

    starts = []

    while len(starts) < num_segments:
        start = np.random.randint(0, total_len - segment_len + 1)

        if all(
            start + segment_len <= s or s + segment_len <= start
            for s in starts
        ):
            starts.append(start)

    indices = []
    for s in starts:
        indices.extend(range(s, s + segment_len))

    return np.array(indices)

subject_path = r'data/model_input'
subject_list = os.listdir(subject_path)
subject_list.sort()

BS = 25

for subject in subject_list:
    with np.load(os.path.join(subject_path, subject)) as data:
        eeg = data['eeg']
        fnirs = data['fnirs']
        label = data['label']

    fnirs *= 1e3

    label = label.astype(float)

    FOLD = 3
    for session in range(FOLD):
        all_eeg = np.delete(eeg, slice(session*(600//FOLD), (session+1)*(600//FOLD)), 0)
        all_fnirs = np.delete(fnirs, slice(session*(600//FOLD), (session+1)*(600//FOLD)), 0)
        all_label = np.delete(label, slice(session*(600//FOLD), (session+1)*(600//FOLD)), 0)

        second_train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": all_eeg, "fnirs_input": all_fnirs},
                {"class_output": all_label, 'eeg_output':all_label}
            )
        ) 
        second_train_dataset = second_train_dataset.shuffle(buffer_size=600, reshuffle_each_iteration=True).batch(BS)

        eeg_test = eeg[session*(600//FOLD):(session+1)*(600//FOLD),]
        fnirs_test = fnirs[session*(600//FOLD):(session+1)*(600//FOLD),]
        label_test = label[session*(600//FOLD):(session+1)*(600//FOLD),]

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": eeg_test, "fnirs_input": fnirs_test},
                {"class_output": label_test, 'eeg_output':label_test} 
            )
        ) 
        test_dataset = test_dataset.batch(BS)

        np.random.seed(42 + session)
        indices = sample_segments(all_eeg.shape[0], 25, 4)

        eeg_train = np.delete(all_eeg, indices, axis=0)
        fnirs_train = np.delete(all_fnirs, indices, axis=0)
        label_train = np.delete(all_label, indices, axis=0)
        first_train_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    {"eeg_input": eeg_train, "fnirs_input": fnirs_train},
                    {"class_output": label_train, 'eeg_output':label_train} 
                )
            ) 
        first_train_dataset = first_train_dataset.shuffle(buffer_size=600, reshuffle_each_iteration=True).batch(BS)

        eeg_val = all_eeg[indices]
        fnirs_val = all_fnirs[indices]
        label_val = all_label[indices]
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": eeg_val, "fnirs_input": fnirs_val},
                {"class_output": label_val, 'eeg_output':label_val} 
            )
        ) 
        val_dataset = val_dataset.batch(BS)

        # print('eeg_train shape:', eeg_train.shape)
        # print('fnirs_train shape:', fnirs_train.shape)
        # print('label_train shape:', label_train.shape)

        # print('eeg_val shape:', eeg_val.shape)
        # print('fnirs_val shape:', fnirs_val.shape)
        # print('label_val shape:', label_val.shape)

        tf.keras.backend.clear_session()
        model = sta_net()

        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-3,
            decay_steps=12 * 500,
            alpha=0.1,
            warmup_target=1e-3,
            warmup_steps=12 * 3,
        )
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-4,
            clipnorm=0.5,
        )
        model.compile(
            optimizer=optimizer,
            loss={
                "class_output": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                "eeg_output": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
            },
            loss_weights={
                "class_output": 1.0,
                "eeg_output": 1.0
            },
            metrics={
                "class_output": "accuracy",
                "eeg_output": "accuracy"
            }
        )

        # stopping = tf.keras.callbacks.EarlyStopping(monitor='val_class_output_loss', patience=50, restore_best_weights=True, verbose=1)
        # stopping = tf.keras.callbacks.EarlyStopping(
        #     monitor='val_class_output_loss',
        #     patience=100,
        #     restore_best_weights=True,
        #     mode='min',
        #     verbose=1
        # )

        print(f"# subject {subject}, session {session}, stage 1")
        plateau_avg = PlateauAveraging(
            monitor="val_class_output_loss",
            window=20,
            # min_delta=1e-3,
            patience=40,
            offset=20,
            trim_ratio=0.25,
        )
        first_history = model.fit(first_train_dataset, epochs = 300,
                verbose = 2, validation_data=val_dataset,
                callbacks=[plateau_avg])
        
        # min_val_class_output_loss = min(first_history.history['val_class_output_loss'])
        # min_val_class_output_loss_epoch = first_history.history['val_class_output_loss'].index(min_val_class_output_loss)
        # # target_acc = first_history.history['class_output_loss'][min_val_class_output_loss_epoch]

        # print(f"# subject {subject}, session {session}, stage 2")
        # # plateau_epochs = plateau_avg.plateau_epochs
        # # if len(plateau_epochs) > 0:
        # #     plateau_start = plateau_epochs[0]
        # #     plateau_end = plateau_epochs[-1]
        # #     best_epoch = int((plateau_start + plateau_end) / 2)
        # # else:
        # #     val_loss = np.array(first_history.history['val_class_output_loss'])
        # #     window = 9
        # #     smooth = np.convolve(val_loss, np.ones(window)/window, mode='valid')
        # #     best_epoch = int(np.argmin(smooth) + window)
        # # print(f"; stage2 epoch = {best_epoch}")
        # stage2_cb = TrainPlateauSWA(
        #     monitor="class_output_accuracy",
        #     patience=10,   # n
        #     swa_k=8,      # k
        #     offset=2,
        # )
        # model.fit(second_train_dataset, epochs = 200,
        #         verbose = 2, validation_data=test_dataset, callbacks=[stage2_cb])
        
        # print('begin test')
        # test_results = model.evaluate(test_dataset)

        # print('begin test')
        test_results = model.evaluate(test_dataset, verbose=0, return_dict=True)

        output_file = "results.txt"

        with open(output_file, "a") as f:
            f.write(f'{{"subject": "{subject}", "fold": {session}, "result": {{')

            for name, value in test_results.items():
                f.write(f"\"{name}\": {value:.6f}, ")

            f.write(f"}}, }}\n")

# print('all done')
