######author: Xiaopeng Lin#######
######This script is used to generate the h5 file for the seet eye tracking dataset######
import os
import h5py
import numpy as np
import tqdm
import tonic
import tonic.transforms as transforms

def process_labels(line):
    return [float(line[0]) / 640, float(line[1]) / 480]

def process_h5_and_labels(file_path, label_path):
    with h5py.File(file_path, 'r') as f:
        x = f['events'][:,1]
        y = f['events'][:,2]
        t = f['events'][:,0]
        p = f['events'][:,3]
        frame_ts = f['frame_ts'][:]
    with open(label_path, 'r') as file:
        labels = np.array([process_labels(line.split()) for line in file], dtype=float)

    samples = []
    sample_labels = []
    segments = []
    segment_index = 0

    for i in range(len(frame_ts)-1):
        segment_start = frame_ts[i]
        segment_end = frame_ts[i+1]
        mask = (t >= segment_start) & (t < segment_end)
        if mask.any():
            segments.append((x[mask], y[mask], t[mask],p[mask]))


    for i, (segment_x, segment_y, segment_t, segment_p) in enumerate(segments):
        if i > len(labels) - 1:
            break
        expand_x = segment_x
        expand_y = segment_y
        expand_t = segment_t
        expand_p = segment_p

        sort_indices = np.argsort(expand_t)
        sorted_x = expand_x[sort_indices]
        sorted_y = expand_y[sort_indices]
        sorted_t = expand_t[sort_indices]
        sorted_p = expand_p[sort_indices]
        if segment_index < len(labels):
            samples.append((sorted_x, sorted_y, sorted_t,sorted_p))
            # print(samples)
        if i < len(labels):
            sample_labels.append(labels[i])
        segment_index += 1

    return samples, sample_labels

def transform(samples,labels,group_size,step_size):
    frames = []
    for sample in samples:
        dtype = [('x', '<i8'), ('y', '<i8'), ('t', '<i8'), ('p', '<i8')]
        events_tonic = np.zeros(len(sample[0]), dtype=dtype)
        events_tonic['x'] = sample[0]/3
        events_tonic['y'] = sample[1]/3
        events_tonic['t'] = sample[2]
        events_tonic['p'] = sample[3]

        transform = tonic.transforms.Compose(
            [
                transforms.ToFrame(
                    sensor_size=(80,60,2),
                    n_time_bins=4,
                ),
                transforms.ToBinaRep(n_frames=1, n_bits=4),
            ]
        )
        frame = transform(events_tonic)
        frames.append(frame)

    frames = np.array(frames)
    labels = np.array(labels)
    print(frames.shape)
    print(labels.shape)
    label_groups = []
    frames_groups = []
    if group_size != 1:
        length = len(frames)//step_size + 1
        number_test = length*step_size 
        frames = np.concatenate((frames, np.repeat(frames[-1:], number_test - len(frames), axis=0)), axis=0)
        labels = np.concatenate((labels, np.repeat(labels[-1:], number_test - len(labels), axis=0)), axis=0)
    for start_idx in range(0, len(frames) - group_size + 1, step_size):
        end_idx = start_idx + group_size
        label_groups.append(labels[start_idx:end_idx])
        frames_groups.append(frames[start_idx:end_idx])
    label_groups = np.array(label_groups)
    frames_groups = np.array(frames_groups)
    print(label_groups.shape)
    frames_groups = np.squeeze(frames_groups, axis=2)
    print(frames_groups.shape)
    return frames_groups, label_groups

def load_filenames(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]
    
def get_data(data_dir,group_size,step_size):
    sorted_data_file_paths = sorted(data_dir)
    print(sorted_data_file_paths)
    all_samples = []
    all_labels = []
    cnt = 0
    for file_path in tqdm.tqdm(sorted_data_file_paths):
        label_path = file_path.replace("data", "label").replace(".h5", ".txt")
        samples, labels = process_h5_and_labels(file_path, label_path)
        samples_group,labels_group = transform(samples,labels,group_size,step_size)
        if cnt == 0:
            frames = samples_group
            label_g = labels_group
        else:
            frames = np.concatenate((frames,samples_group),axis=0)
            label_g = np.concatenate((label_g,labels_group),axis=0)
        cnt += 1
    return frames,label_g

root_dir = '/mnt/e/DS/eye_tracking/'
trainfile = "/mnt/e/DS/eye_tracking/train.txt"
testfile = "/mnt/e/DS/eye_tracking/test.txt"
group_size = 100
step_size = 100
train_filenames = load_filenames(trainfile)
val_filenames = load_filenames(testfile)
data_train = [os.path.join(root_dir+'data//', f + '.h5') for f in train_filenames]
data_val = [os.path.join(root_dir+'data//', f + '.h5') for f in val_filenames]
data_train,label_train = get_data(data_train,group_size,step_size)
data_test, label_test = get_data(data_val,group_size,step_size)

print('len(all_samples)=', len(data_train))
print('len(all_labels)=', len(data_test))

X_train = data_train
y_train = label_train
X_test = data_test
y_test = label_test

EXPORT_PATH = './'
with h5py.File(os.path.join(EXPORT_PATH,"train_3et.h5"), 'a') as hf:
    fset = hf.create_dataset('frames', shape=X_train.shape, maxshape = (None,group_size, 2,60,80), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=y_train.shape, maxshape = (None,group_size,3), chunks=True, dtype='float32')
    hf['frames'][:] = X_train
    hf['label'][:] = y_train

with h5py.File(os.path.join(EXPORT_PATH,"test_3et.h5"), 'a') as hf:
    fset = hf.create_dataset('frames', shape=X_test.shape, maxshape = (None,group_size, 2,60,80), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=y_test.shape, maxshape = (None,group_size,3), chunks=True, dtype='float32')
    hf['frames'][:] = X_test
    hf['label'][:] = y_test