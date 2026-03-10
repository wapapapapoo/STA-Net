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


subject_path = r'data/model_input'
subject_list = os.listdir(subject_path)
subject_list.sort()

BS = 32

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
        indices = np.random.choice(all_eeg.shape[0], size=80, replace=False)

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

        print(f"# subject {subject}, session {session}")

        tf.keras.backend.clear_session()
        model = sta_net()

        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
        # lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        #     boundaries=[80, 140],
        #     values=[3e-4, 1e-4, 3e-5]
        # )
        # optimizer = tf.keras.optimizers.SGD(
        #     learning_rate=lr_schedule,
        #     momentum=0.9,
        #     nesterov=True,
        #     weight_decay=1e-4,
        #     clipnorm=1.0,
        # )
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=5e-3,
            weight_decay=5e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        model.compile(
            optimizer=optimizer,
            loss={
                "class_output": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                "eeg_output": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
            },
            loss_weights={
                "class_output": 1.0,
                "eeg_output": .3
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

        # print('begin first train')
        # first_history = model.fit(first_train_dataset, epochs = 200,
        #         verbose = 2, validation_data=val_dataset,)
        #         #callbacks=[stopping])
        
        # min_val_class_output_loss = min(first_history.history['val_class_output_loss'])
        # min_val_class_output_loss_epoch = first_history.history['val_class_output_loss'].index(min_val_class_output_loss)
        # target_acc = first_history.history['class_output_loss'][min_val_class_output_loss_epoch]

        # print('begin second train')
        # best_epoch = min_val_class_output_loss_epoch + 1 
        model.fit(second_train_dataset, epochs = 200,
                verbose = 2, validation_data=test_dataset)#callbacks=[targetacccallback(target_acc)])
        
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
