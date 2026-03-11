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
        window=5,
        min_delta=1e-3,
        patience=20
    ):
        super().__init__()

        self.monitor = monitor
        self.window = window
        self.min_delta = min_delta
        self.patience = patience

        self.loss_history = []
        self.swa_weights = None
        self.n_models = 0

        self.plateau_started = False
        self.wait = 0

    def moving_avg(self):
        arr = np.array(self.loss_history)
        if len(arr) < self.window:
            return None
        return np.mean(arr[-self.window:])

    def on_epoch_end(self, epoch, logs):

        val_loss = logs[self.monitor]
        self.loss_history.append(val_loss)

        smooth = self.moving_avg()
        if smooth is None:
            return

        # 判断是否进入 plateau
        if len(self.loss_history) > self.window:
            prev = np.mean(self.loss_history[-self.window-1:-1])
            improvement = prev - smooth

            if abs(improvement) < self.min_delta:
                if not self.plateau_started:
                    print(f"Plateau detected at epoch {epoch+1}")
                    self.plateau_started = True

        # plateau 内开始平均权重
        if self.plateau_started:

            weights = self.model.get_weights()

            if self.swa_weights is None:
                self.swa_weights = [w.copy() for w in weights]
            else:
                for i in range(len(weights)):
                    self.swa_weights[i] = (
                        self.swa_weights[i] * self.n_models + weights[i]
                    ) / (self.n_models + 1)

            self.n_models += 1

            self.wait += 1

            if self.wait >= self.patience:
                print("Plateau stable → stopping training")
                self.model.stop_training = True

    def on_train_end(self, logs=None):

        if self.swa_weights is not None:
            print(f"Applying plateau-averaged weights ({self.n_models} models)")
            self.model.set_weights(self.swa_weights)

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

BS = 4

for subject in subject_list:
    with np.load(os.path.join(subject_path, subject)) as data:
        eeg = data['eeg']
        fnirs = data['fnirs']
        label = data['label']

    fnirs *= 1e3

    label = label.astype(float)

    FOLD = 6
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
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=3e-4,
            momentum=0.9,
            nesterov=True,
            weight_decay=1e-4,
            clipnorm=1.0,
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
        stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_class_output_loss',
            patience=50,
            restore_best_weights=True,
            mode='min',
            verbose=1
        )

        print(f"# subject {subject}, session {session}, stage 1")
        plateau_avg = PlateauAveraging(
            monitor="val_class_output_loss",
            window=5,
            min_delta=1e-3,
            patience=20
        )
        first_history = model.fit(first_train_dataset, epochs = 300,
                verbose = 2, validation_data=val_dataset,
                callbacks=[plateau_avg])
        
        min_val_class_output_loss = min(first_history.history['val_class_output_loss'])
        min_val_class_output_loss_epoch = first_history.history['val_class_output_loss'].index(min_val_class_output_loss)
        # target_acc = first_history.history['class_output_loss'][min_val_class_output_loss_epoch]

        print(f"# subject {subject}, session {session}, stage 2")
        val_loss = np.array(first_history.history['val_class_output_loss'])
        window = 5
        smooth = np.convolve(val_loss, np.ones(window)/window, mode='valid')
        best_epoch = int(np.argmin(smooth) + window)
        model.fit(second_train_dataset, epochs = best_epoch,
                verbose = 2, validation_data=test_dataset)
        
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
