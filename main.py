import h5py
import numpy as np
import tensorflow as tf
from pprint import pprint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense, Dropout
from eval import evaluate_model
import pickle
import os


files_train = []
labels_train = []
files_eval = []
labels_eval = []


leave_out_subjects = [{'type': 'MCA', 'id': '0002'}, 
                     {'type': 'TIA', 'id': '0001'}] 

for cls, cls_id in [('MCA', 1), ('TIA', 0)]:
    # folder = os.path.join('data/landmarks', cls)
    folder = os.path.join('data/landmarks_processed/', cls)
    for fn in os.listdir(folder):
        if fn.endswith('.h5'):
            if any(leave_out['type'] == cls and leave_out['id'] in fn for leave_out in leave_out_subjects):
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

window_size = 500
X = []
y = []
for seq, lbl in zip(raw_sequences_train, labels_train):
    n_frames, n_landmarks, n_coordinates = seq.shape
    n_windows = n_frames // window_size
    if n_windows == 0:
        continue  # in case some file is shorter than window_size
    
    # slice into non-overlapping windows
    windowed = seq[: n_windows * window_size] \
                   .reshape(n_windows, window_size, n_landmarks, n_coordinates)
    
    X.append(windowed)
    y.append(np.full(n_windows, lbl, dtype=np.int32))

X = np.concatenate(X, axis=0)       # shape = (total_windows, window_size, n_landmarks, n_coordinates)
y = np.concatenate(y, axis=0)       # shape = (total_windows,)
X = X[..., np.newaxis].astype('float32')

X_eval = []
y_eval = []
for seq, lbl in zip(raw_sequences_eval, labels_eval):
    n_frames, n_landmarks, n_coordinates = seq.shape
    n_windows = n_frames // window_size
    if n_windows == 0:
        continue
    
    windowed = seq[: n_windows * window_size] \
                   .reshape(n_windows, window_size, n_landmarks, n_coordinates)
    
    X_eval.append(windowed)
    y_eval.append(np.full(n_windows, lbl, dtype=np.int32))

X_eval = np.concatenate(X_eval, axis=0)
y_eval = np.concatenate(y_eval, axis=0)
X_eval = X_eval[..., np.newaxis].astype('float32')

pprint(f'X shape: {X.shape}, y shape: {y.shape}')

# 4. One-hot your labels if needed
num_classes = len(np.unique(y))

pprint(f'Number of classes: {num_classes}')

# 6. Build the ConvLSTM2D model
model = Sequential([
    ConvLSTM2D(
        filters=16,
        kernel_size=(3,3),
        padding='same',
        return_sequences=False,
        input_shape=(window_size, n_landmarks, n_coordinates, 1)
    ),
    BatchNormalization(),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=[
        'accuracy', 
        tf.keras.metrics.Precision(name='precision'), 
        tf.keras.metrics.Recall(name='recall')
    ]
)

model.summary()

# 7. Train
history = model.fit(
    X, y,
    validation_data=(X_eval, y_eval),
    epochs=20,
    batch_size=8
)

results = model.evaluate(X_eval, y_eval, verbose=1)
results_dict = dict(zip(model.metrics_names, results))
pprint(results_dict)

# 8. Save the model
leaveout_names = [f"{leave_out['type']}_{leave_out['id']}" for leave_out in leave_out_subjects]
leaveout_names = '_'.join(leaveout_names)
name = f'conv_lstm_model__leaveout_{leaveout_names}'
model.save(os.path.join('models', name + '.keras'))



leave_out_file = os.path.join('models', name + '_eval.h5')
with h5py.File(leave_out_file, 'w') as f:
    f.create_dataset('pose_landmarks', data=X_eval)
    f.attrs['labels'] = y_eval.tolist()

# 9. Evaluate the model

evaluate_model(model, X_eval, y_eval)