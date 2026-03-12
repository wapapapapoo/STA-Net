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
        min_delta=0,
        patience=20,
        trim_ratio=0.2
    ):
        super().__init__()

        self.monitor = monitor
        self.window = window
        self.min_delta = min_delta
        self.patience = patience
        self.trim_ratio = trim_ratio

        self.loss_history = []

        self.plateau_started = False
        self.wait = 0

        # 保存 plateau 阶段的 (loss, weights)
        self.plateau_records = []

    def trimmed_mean(self, arr):
        arr = np.sort(arr)
        k = int(len(arr) * self.trim_ratio)
        if k > 0:
            arr = arr[:-k]
        return np.mean(arr)

    def moving_avg(self):
        if len(self.loss_history) < self.window:
            return None
        return self.trimmed_mean(self.loss_history[-self.window:])

    def on_epoch_end(self, epoch, logs=None):

        val_loss = logs[self.monitor]
        self.loss_history.append(val_loss)

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

            weights = self.model.get_weights()
            self.plateau_records.append(
                (val_loss, [w.copy() for w in weights])
            )

            self.wait += 1

            if self.wait >= self.patience:
                print("; Plateau stable → stopping")
                self.model.stop_training = True

    def on_train_end(self, logs=None):

        if len(self.plateau_records) == 0:
            return

        records = self.plateau_records

        losses = np.array([r[0] for r in records])

        k = int(len(losses) * self.trim_ratio)

        # 根据loss排序
        idx = np.argsort(losses)

        # 去掉最大的k个
        if k > 0:
            idx = idx[:-k]

        selected = [records[i][1] for i in idx]

        print(f"; Trimmed SWA using {len(selected)} / {len(records)} epochs")

        avg_weights = []

        for ws in zip(*selected):
            avg_weights.append(np.mean(ws, axis=0))

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
    possible_starts = list(range(0, total_len - segment_len + 1, 10))
    while len(starts) < num_segments:
        start = np.random.choice(possible_starts)
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

        np.random.seed(42)
        indices = sample_segments(all_eeg.shape[0], 20, 5)

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















        retry_bar = 0.8
        best_acc = -1
        best_weights = None

        for i in range(20):
            seed = np.random.randint(0, 114514)
            np.random.seed(seed)
            tf.random.set_seed(seed)

            tf.keras.backend.clear_session()
            model = sta_net()

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

            print(f"# subject {subject}, session {session}, stage 1, retry {i}")

            plateau_avg = PlateauAveraging(
                monitor="val_class_output_loss",
                window=20,
                patience=20,
                trim_ratio=0.25,
            )

            first_history = model.fit(
                first_train_dataset,
                epochs=300,
                verbose=2,
                validation_data=val_dataset,
                callbacks=[plateau_avg]
            )

            val_result = model.evaluate(val_dataset, verbose=0, return_dict=True)
            cur_acc = val_result["class_output_accuracy"]

            print("; stage1 final val acc:", cur_acc)

            if cur_acc > best_acc:
                best_acc = cur_acc
                best_weights = model.get_weights()

            if cur_acc >= retry_bar:
                print(f"; reached retry_bar {retry_bar}, stop retry")
                break


        # 如果没有达到 bar，则恢复最佳模型
        if best_weights is not None:
            model.set_weights(best_weights)

        print("; stage1 final val acc:", best_acc)







            
        min_val_class_output_loss = min(first_history.history['val_class_output_loss'])
        min_val_class_output_loss_epoch = first_history.history['val_class_output_loss'].index(min_val_class_output_loss)
        # target_acc = first_history.history['class_output_loss'][min_val_class_output_loss_epoch]

        print(f"# subject {subject}, session {session}, stage 2")
        stage2_cb = TrainPlateauSWA(
            monitor="class_output_accuracy",
            patience=10,   # n
            swa_k=8,      # k
            offset=2,
        )
        model.fit(second_train_dataset, epochs = 200,
                verbose = 2, validation_data=test_dataset, callbacks=[stage2_cb])








        test_results = model.evaluate(test_dataset, verbose=0, return_dict=True)

        output_file = "results.txt"

        with open(output_file, "a") as f:
            f.write(f'{{"subject": "{subject}", "fold": {session}, "result": {{')

            for name, value in test_results.items():
                f.write(f"\"{name}\": {value:.6f}, ")

            f.write(f"}}, }}\n")

# print('all done')
