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
    loss, accuracy = model.evaluate(X_eval, y_eval, verbose=1)
    
    print(f"Evaluation Loss: {loss}")
    print(f"Evaluation Accuracy: {accuracy}")


if __name__ == "__main__":
    
    leave_out_file = 'models/conv_lstm_model__leaveout_MCA_0002.pkl'

    if not os.path.exists(leave_out_file):
        raise FileNotFoundError(f"Model file {leave_out_file} does not exist.")
    with open(leave_out_file, 'rb') as f:
        leave_out = pickle.load(f)

    X = leave_out['sequence']
    y = [types[leave_out['type']]] * len(X)

    model = load_model('models/conv_lstm_model__leaveout_MCA_0002.keras')

    # Evaluate the model
    evaluate_model(model, X, y)
