import h5py
import numpy as np
import tensorflow as tf
from pprint import pprint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense


import os

files = []
labels = []

for cls, cls_id in [('MCA', 1), ('TIA', 0)]:
    folder = os.path.join('data/landmarks', cls)
    for fn in os.listdir(folder):
        if fn.endswith('.h5'):
            files.append(os.path.join(folder, fn))
            labels.append(cls_id)

files = np.array(files)
leave_one_subject_out = True

labels = np.array(labels, dtype=np.int32)

raw_sequences = [h5py.File(fn)['pose_landmarks'] for fn in files ]

print(f'Found {len(raw_sequences)} sequences with {len(files)} files.')

window_size = 30
X = []
y = []
for seq in raw_sequences:
    n_frames, n_landmarks, n_coordinates = seq.shape
    n_windows = n_frames // window_size
    windowed_sequences = seq[:n_windows*window_size].reshape(n_windows, window_size, n_landmarks, n_coordinates)
    
    y.append(np.full(n_windows, labels[0]))  #all frames in a sequence have the same label
    X.append(windowed_sequences)  

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)
# 3. Add channel dim → (batch, time, H, W, 1)
X = X[..., np.newaxis].astype('float32')

pprint(f'X shape: {X.shape}, y shape: {y.shape}')

# 4. One-hot your labels if needed
num_classes = len(np.unique(y))

# 5. Split train/test
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

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
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# 7. Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=8
)

# 8. Save the model
model.save('models/conv_lstm_model.keras')