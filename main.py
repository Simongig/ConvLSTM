from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf
from pprint import pprint
from tensorflow import keras
from keras.models import Sequential
from keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense, Dropout, LSTM, Input
from keras.optimizers import Adam
from eval import evaluate_model
import os
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import pandas as pd

# settings
config = {
    "sample_size": 500,  # number of frames in each sample
    "lstm1_units": 32,
    "conv_lstm_num_filters": 32,
    "dropout_rate": 0.4,
    "learning_rate": 0.001,
    "metrics": ['accuracy', 'Precision', 'Recall', 'AUC'],

    # fit settings
    "epochs": 20,
    "batch_size": 8
}

leave_out_subjects = [{'type': 'MCA', 'id': '0002'}, 
                     {'type': 'TIA', 'id': '0001'}] 

def createLSTMModel(input_shape, num_classes, lstm1_units=64, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(
            units=lstm1_units,
            return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=16, activation='relu'),
        Dense(units=1, activation='sigmoid'),
    ])

    # optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=config['metrics']
    )
    return model

def createConvLSTM(input_shape, num_classes, num_filters=32, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        Input(shape=input_shape),
        ConvLSTM2D(
            filters=num_filters,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=False),
        Dropout(dropout_rate),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')     
    ])

    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=config['metrics']
    )
    return model

def loadAndSplitData(leave_out_subjects):    
    files_train = []
    labels_train = []
    files_eval = []
    labels_eval = []

    for cls, cls_id in [('MCA', 1), ('TIA', 0)]:
        folder = os.path.join('data/landmarks_processed/', cls)
        for fn in os.listdir(folder):
            if fn.endswith('.h5'):
                is_leaveout = any(leave_out['type'] == cls and leave_out['id'] in fn for leave_out in leave_out_subjects)
                if is_leaveout:
                    files_eval.append(os.path.join(folder, fn))
                    labels_eval.append(cls_id)
                else:
                    files_train.append(os.path.join(folder, fn))
                    labels_train.append(cls_id)

    files_train = np.array(files_train)
    files_eval = np.array(files_eval)

    labels_eval = np.array(labels_eval, dtype=np.int32)
    labels_train = np.array(labels_train, dtype=np.int32)

    pprint(f'Number of training files: {len(files_train)}')
    pprint(f'Number of evaluation files: {len(files_eval)}')

    raw_sequences_train = [h5py.File(fn)['pose_landmarks'] for fn in files_train ]
    raw_sequences_eval = [h5py.File(fn)['pose_landmarks'] for fn in files_eval ]

    return raw_sequences_train, raw_sequences_eval, labels_train, labels_eval

def reshapeSequencesForLSTM(data_sequence, labels):
    X_windows = []
    y_windows = []
    for seq, lbl in zip(data_sequence, labels):
        n_frames, n_landmarks, n_coordinates = seq.shape
        n_windows = n_frames // config['sample_size']
        if n_windows == 0:
            continue  # in case some file is shorter than window_size
        
        # slice into non-overlapping windows
        windowed = seq[: n_windows * config['sample_size']] \
                    .reshape(n_windows, config['sample_size'], n_landmarks, n_coordinates)
        
        X_windows.append(windowed)
        y_windows.append(np.full(n_windows, lbl, dtype=np.int32))


    # Concatenate all windows into a single array
    X = np.concatenate(X_windows, axis=0) 
    y = np.concatenate(y_windows, axis=0)

    return X, y

def saveModelAndData(model, X_eval, y_eval, prefix="lstm_model"):
    leaveout_names = [f"{leave_out['type']}_{leave_out['id']}" for leave_out in leave_out_subjects]
    leaveout_names = '_'.join(leaveout_names)

    name = f'{prefix}__leaveout_{leaveout_names}'
    model.save(os.path.join('models', name + '.keras'))

    leave_out_file = os.path.join('models', name + '_eval.h5')
    with h5py.File(leave_out_file, 'w') as f:
        f.create_dataset('pose_landmarks', data=X_eval)
        f.attrs['labels'] = y_eval.tolist()

def appendResultsToDataFrame(results_dict, csv_path='results/experiment_log.csv'):
    path = Path(csv_path)
    path.parent.mkdir(exist_ok=True)
    
    # Konvertiere das Dictionary in ein DataFrame
    df = pd.DataFrame([results_dict])
    
    # Schreibe in die CSV-Datei
    # mode='a' zum Anhängen, header=not path.exists() schreibt den Header nur, wenn die Datei neu ist
    df.to_csv(path, mode='a', header=not path.exists(), index=False)
    print(f"Ergebnisse erfolgreich in {csv_path} geloggt.")


def main():
    raw_sequences_train, raw_sequences_eval, labels_train, labels_eval  = loadAndSplitData(leave_out_subjects)

    X, y = reshapeSequencesForLSTM(raw_sequences_train, labels_train)
    X_eval, y_eval = reshapeSequencesForLSTM(raw_sequences_eval, labels_eval)

    unique_classes = len(np.unique(labels_train))

    ### 1. LSTM model ###
    X_lstm = X.reshape(X.shape[0], X.shape[1], X.shape[2] * X.shape[3])  # shape = (total_windows, window_size, n_landmarks * n_coordinates)
    X_eval_lstm = X_eval.reshape(X_eval.shape[0], X_eval.shape[1], X_eval.shape[2] * X_eval.shape[3])  # shape = (total_windows, window_size, n_landmarks * n_coordinates)

    pprint(f'LSTM: X shape: {X_lstm.shape}, y shape: {y.shape}')

    lstm_model = createLSTMModel(
        input_shape=(X_lstm.shape[1], X_lstm.shape[2]),  # (window_size, n_landmarks * n_coordinates)
        num_classes=unique_classes,
        lstm1_units=config['lstm1_units'],
        dropout_rate=config['dropout_rate'],
        learning_rate=config['learning_rate']
    )
    lstm_model.summary()
    history_lstm = lstm_model.fit(
        X_lstm, y,
        validation_data=(X_eval_lstm, y_eval),
        epochs=20,
        batch_size=8
    )
    
    saveModelAndData(lstm_model, X_eval_lstm, y_eval, prefix="compare_lstm_model")
    metrics_dict_lstm = evaluate_model(lstm_model, X_eval_lstm, y_eval)
    appendResultsToDataFrame(metrics_dict_lstm)

    ### 2. ConvLSTM model ###
    X_conv_lstm = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1)  # Add channel dimension
    X_eval_conv_lstm = X_eval.reshape(X_eval.shape[0], X_eval.shape[1], X_eval.shape[2], X_eval.shape[3], 1)  # Add channel dimension

    pprint(f'ConvLSTM: X shape: {X_conv_lstm.shape}, y shape: {y.shape}')

    conv_lstm_model = createConvLSTM(
        input_shape=(X_conv_lstm.shape[1], X_conv_lstm.shape[2], X_conv_lstm.shape[3], 1),
        num_classes=unique_classes,
        num_filters=config['conv_lstm_num_filters'],
        dropout_rate=config['dropout_rate'],
        learning_rate=config['learning_rate']
        )
    conv_lstm_model.summary()
    history_conv_lstm = conv_lstm_model.fit(
       X_conv_lstm, y,  # Add channel dimension
        validation_data=(X_eval_conv_lstm, y_eval),
        epochs=20,
        batch_size=8
    )

    saveModelAndData(conv_lstm_model, X_eval_conv_lstm, y_eval, prefix="compare_conv_lstm_model")
    metrics_dict_conv_lstm = evaluate_model(conv_lstm_model, X_eval_conv_lstm, y_eval)
    appendResultsToDataFrame(metrics_dict_conv_lstm)

if __name__ == '__main__':
    main()