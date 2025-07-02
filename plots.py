import numpy as np
from sklearn.metrics import roc_curve, auc
from keras.models import load_model
import matplotlib.pyplot as plt
import h5py

# load X und y from h5 file
h5py_file = h5py.File('models/compare_conv_lstm_model__leaveout_MCA_0002_TIA_0001_eval.h5', 'r')
X_test = h5py_file['pose_landmarks'][:]
y_test = h5py_file.attrs['labels'].tolist()


# Annahme: Sie haben Ihr Keras LSTM-Modell bereits trainiert.
model = load_model('models/compare_conv_lstm_model__leaveout_MCA_0002_TIA_0001.keras')
# X_test = Ihre Testdaten
# y_test = Ihre wahren Labels für die Testdaten (0 oder 1)

# Erhalten Sie die vorhergesagten Wahrscheinlichkeiten für die positive Klasse
y_pred_lstm = model.predict(X_test).ravel()

print("LSTM Predictions:", y_pred_lstm)
print("True Labels:     ", y_test)


# 2. ROC-Werte berechnen
fpr, tpr, thresholds = roc_curve(y_test, y_pred_lstm)

# 3. AUC berechnen
roc_auc = auc(fpr, tpr)

# 4. Kurve plotten
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f'LSTM ROC curve (area = {roc_auc:.2f})', color="orange")
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve für ConvLSTM')
plt.legend(loc="lower right")
plt.show()

# confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Berechnung der Konfusionsmatrix
cm = confusion_matrix(y_test, (y_pred_lstm > 0.5).astype(int))
# Plotten der Konfusionsmatrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['TIA', 'MCA'], yticklabels=['TIA', 'MCA'])
plt.xlabel('Vorhergesagte Labels')
plt.ylabel('Wahre Labels')
plt.title('Konfusionsmatrix für ConvLSTM')
plt.show()