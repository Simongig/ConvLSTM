import h5py
import numpy as np
import tensorflow as tf
from pprint import pprint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense
from eval import evaluate_model
import pickle
import os


files_train = []
labels_train = []
files_eval = []
labels_eval = []


leave_out_subject = {'type': 'MCA', 'id': '0002'}

for cls, cls_id in [('MCA', 1), ('TIA', 0)]:
    folder = os.path.join('data/landmarks', cls)
    for fn in os.listdir(folder):
        if fn.endswith('.h5'):
            if cls == leave_out_subject['type'] and fn.startswith(leave_out_subject['id']):
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

window_size = 250
X = []
y = []
for seq in raw_sequences_train:
    n_frames, n_landmarks, n_coordinates = seq.shape
    n_windows = n_frames // window_size
    windowed_sequences = seq[:n_windows*window_size].reshape(n_windows, window_size, n_landmarks, n_coordinates)
    
    y.append(np.full(n_windows, labels_train[0]))  #all frames in a sequence have the same label
    X.append(windowed_sequences)  

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)
# 3. Add channel dim → (batch, time, H, W, 1)
X = X[..., np.newaxis].astype('float32')

X_eval = []
y_eval = []

for seq in raw_sequences_eval:
    n_frames, n_landmarks, n_coordinates = seq.shape
    n_windows = n_frames // window_size
    windowed_sequences = seq[:n_windows*window_size].reshape(n_windows, window_size, n_landmarks, n_coordinates)
    
    y_eval.append(np.full(n_windows, labels_eval[0]))  #all frames in a sequence have the same label
    X_eval.append(windowed_sequences)

X_eval = np.concatenate(X_eval, axis=0)
y_eval = np.concatenate(y_eval, axis=0)
X_eval = X_eval[..., np.newaxis].astype('float32')

pprint(f'X shape: {X.shape}, y shape: {y.shape}')

# 4. One-hot your labels if needed
num_classes = len(np.unique(y))

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
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# 7. Train
history = model.fit(
    X, y,
    validation_data=(X_eval, y_eval),
    epochs=20,
    batch_size=8
)

# 8. Save the model
name = f'conv_lstm_model__leaveout_{leave_out_subject["type"]}_{leave_out_subject["id"]}'
model.save(os.path.join('models', name + '.keras'))

# save leavout subject sequence
leave_out_subject_sequence = {
    'type': leave_out_subject['type'],
    'id': leave_out_subject['id'],
    'sequence': raw_sequences_eval
}

with open(os.path.join('models', name + '.pkl'), 'wb') as f:
    pickle.dump(leave_out_subject_sequence, f)

# 9. Evaluate the model

evaluate_model(model, X_eval, y_eval)