from sta import sta_net

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
import os


class targetacccallback(keras.callbacks.Callback):
    def __init__(self, target_acc):
        super().__init__()

        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs={}):
        if(logs['class_output_loss'] <= self.target_acc):
            print("\nReached target loss value {} so cancelling training!\n".format(self.target_acc))
            self.model.stop_training = True


subject_path = r'data/model_input'
subject_list = os.listdir(subject_path)

for subject in subject_list:
    with np.load(os.path.join(subject_path, subject)) as data:
        eeg = data['eeg']
        fnirs = data['fnirs']
        label = data['label']

    fnirs *= 1e3

    label = label.astype(float)

    for session in range(3):
        all_eeg = np.delete(eeg, slice(session*200, (session+1)*200), 0)
        all_fnirs = np.delete(fnirs, slice(session*200, (session+1)*200), 0)
        all_label = np.delete(label, slice(session*200, (session+1)*200), 0)

        second_train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": all_eeg, "fnirs_input": all_fnirs},
                {"class_output": all_label, 'eeg_output':all_label}
            )
        ) 
        second_train_dataset = second_train_dataset.shuffle(buffer_size=128).batch(32)

        eeg_test = eeg[session*200:(session+1)*200,]
        fnirs_test = fnirs[session*200:(session+1)*200,]
        label_test = label[session*200:(session+1)*200,]

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": eeg_test, "fnirs_input": fnirs_test},
                {"class_output": label_test, 'eeg_output':label_test} 
            )
        ) 
        test_dataset = test_dataset.batch(32)

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
        first_train_dataset = first_train_dataset.shuffle(buffer_size=128).batch(32)

        eeg_val = all_eeg[indices]
        fnirs_val = all_fnirs[indices]
        label_val = all_label[indices]
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {"eeg_input": eeg_val, "fnirs_input": fnirs_val},
                {"class_output": label_val, 'eeg_output':label_val} 
            )
        ) 
        val_dataset = val_dataset.batch(32)

        print('eeg_train shape:', eeg_train.shape)
        print('fnirs_train shape:', fnirs_train.shape)
        print('label_train shape:', label_train.shape)

        print('eeg_val shape:', eeg_val.shape)
        print('fnirs_val shape:', fnirs_val.shape)
        print('label_val shape:', label_val.shape)

        print(subject)
        print(session)

        tf.keras.backend.clear_session()
        model = sta_net()

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

        stopping = tf.keras.callbacks.EarlyStopping(monitor='val_class_output_loss', patience=50, restore_best_weights=True, verbose=1)

        print('begin first train')
        first_history = model.fit(first_train_dataset, epochs = 300, 
                verbose = 2, validation_data=val_dataset,
                callbacks=[stopping])
        
        min_val_class_output_loss = min(first_history.history['val_class_output_loss'])
        min_val_class_output_loss_epoch = first_history.history['val_class_output_loss'].index(min_val_class_output_loss)
        target_acc = first_history.history['class_output_loss'][min_val_class_output_loss_epoch]

        print('begin second train')
        model.fit(second_train_dataset, epochs = 200, 
                verbose = 2, callbacks=[targetacccallback(target_acc)])
        
        print('begin test')
        test_results = model.evaluate(test_dataset)
    

print('all done')

            

        