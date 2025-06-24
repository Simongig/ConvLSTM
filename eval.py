import h5py
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from pprint import pprint
import os

types = { 'MCA': 1, 'TIA': 0 }

def evaluate_model(model, X_eval, y_eval):
    """
    Evaluate the model on the evaluation dataset.
    """
    # Convert lists to numpy arrays
    X_eval = np.array(X_eval)
    y_eval = np.array(y_eval)

    # Evaluate the model
    results = model.evaluate(X_eval, y_eval, verbose=1)

    results_dict = dict(zip(model.metrics_names, results))

    print("\nEvaluation Results:")
    for name, value in results_dict.items():
        print(f"{name.capitalize()}: {value:.4f}")
    print("\nConfusion Matrix:")
    y_pred = model.predict(X_eval)
    y_pred_classes = np.argmax(y_pred, axis=1)
    confusion_matrix = tf.math.confusion_matrix(y_eval, y_pred_classes, num_classes=len(types))
    
    print(confusion_matrix.numpy())
    print("\nClassification Report:")
    from sklearn.metrics import classification_report
    report = classification_report(y_eval, y_pred_classes, target_names=list(types.keys()), output_dict=True)
    pprint(report)
    
    return results_dict

if __name__ == "__main__":
    
    leave_out_file = 'models/conv_lstm_model__leaveout_MCA_0002_TIA_0001_eval.h5'

    if not os.path.exists(leave_out_file):
        raise FileNotFoundError(f"Model file {leave_out_file} does not elandmarksist.")
    leave_out_file = h5py.File(leave_out_file, 'r')
    print("Leave-out dataset entry names:", list(leave_out_file.keys()))

    X = leave_out_file['pose_landmarks'][:]
    y = leave_out_file.attrs['labels']
    leave_out_file.close()

    print(f"landmarks shape: {X.shape}")

    model = load_model('models/prototype__conv_lstm_model__leaveout_MCA_0002_TIA_0001.keras')

    # Evaluate the model
    evaluate_model(model, X, y)
