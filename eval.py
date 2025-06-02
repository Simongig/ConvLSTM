import h5py
import numpy as np
import tensorflow as tf
from pprint import pprint
import os


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

