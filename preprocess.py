import h5py
import numpy as np
import os

def remove_landmarks():
    """
    Remove the first 5 landmarks from the dataset.
    """
    files = []
    labels = []

    for cls, cls_id in [('MCA', 1), ('TIA', 0)]:
        folder = f'data/landmarks/{cls}'
        for fn in os.listdir(folder):
            if fn.endswith('.h5'):
                files.append(os.path.join(folder, fn))
                labels.append(cls_id)

    files = np.array(files)
    labels = np.array(labels, dtype=np.int32)

    print(f'Number of files: {len(files)}')

    for fn in files:
        with h5py.File(fn, 'r+') as f:
            seq = f['pose_landmarks'][:]
            
            # Remove aall landmarks except for upper limb
            new_seq = seq[:, 11:22, :]
            fn = fn.replace('\\', '/')
            type = fn.split('/')[-2]
            new_folder = f'data/landmarks_processed/{type}'
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            new_filename = os.path.join(new_folder, os.path.basename(fn))
            
            print(f'Saving file: {new_filename}, shape: {seq.shape}')

            with h5py.File(new_filename, 'w') as new_f:
                new_f.create_dataset('pose_landmarks', data=new_seq)



if __name__ == "__main__":
    remove_landmarks()