import os
import re
import h5py
import numpy as np
import tonic
import tonic.transforms as transforms

def shuffle_downsample(data,num=None):
    ''' data is a numpy array '''
    if num == None:
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
    elif num > data.shape[0]:
        idx = np.random.choice(np.arange(data.shape[0]), size=num, replace=True)
        idx.sort()
    else:
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
        idx = idx[0:num]
        idx.sort()
    return data[idx,...]

def normaliztion(orinal_events,w,h):
    """
    Normalize events.
    """
    events = orinal_events.copy()
    events = events.astype('float32')
    events[:, 0] = (events[:, 0] - events[:, 0].min(axis=0)) / (events[:, 0].max(axis=0) - events[:, 0].min(axis=0)+1e-6)
    events[:, 1] = events[:, 1] / w
    events[:, 2] = events[:, 2] / h
    return events

def process_labels(line):
    numbers = re.sub(r'[()]', '', line).split(',')
    return [float(numbers[0]) / 640, float(numbers[1]) / 480, float(numbers[2])]

def process_labels_test(line):
    numbers = re.sub(r'[()]', '', line).split(',')
    return [float(numbers[0]) / 80, float(numbers[1]) / 60, float(numbers[2])]

def process_h5_and_labels(file_path, label_path,SLIDING,test=False):
    with h5py.File(file_path, 'r') as f:
        x = f['events']['x'][:]
        y = f['events']['y'][:]
        t = f['events']['t'][:]
        p = f['events']['p'][:]

    with open(label_path, 'r') as file:
        if test:
            label_before = np.array([process_labels_test(line.strip()) for line in file], dtype=float)
        else:
            label_before = np.array([process_labels(line.strip()) for line in file], dtype=float)

    labels = []
    for i in range(len(label_before)):
            current_x = label_before[i][0]
            current_y = label_before[i][1]
            if i < len(label_before) - 1:
                next_x = label_before[(i+1)][0]
                next_y = label_before[(i+1)][1]
                if current_x == next_x and current_y == next_y:
                    current_state = 1
                else:
                    current_state = label_before[i][2]
            else:
                current_state = label_before[i][2]
            if label_before[i][2] == 1:
                current_state =1
            labels.append([current_x,current_y,current_state])

    return (t,x,y,p), labels

def process(samples,labels,SLIDING):
    start_time = samples[0][0]
    end_time = samples[0][-1]
    (t,x,y,p) = samples
    samples = []
    sample_labels = []
    segments = []
    sliding_window = SLIDING
    for segment_start in np.arange(start_time, end_time, sliding_window):
        segment_end = segment_start + sliding_window
        mask = (t >= segment_start) & (t < segment_end)
        if segment_end > end_time:
            segment_end = end_time
            mask = (t >= segment_start) & (t <= segment_end)
        segments.append((x[mask], y[mask], t[mask], p[mask]))

    for i, (segment_x, segment_y, segment_t, segment_p) in enumerate(segments):
        if i > len(labels) - 1:
            break
        expand_x = segment_x/640
        expand_y = segment_y/480
        expand_t = segment_t
        expand_p = segment_p
        sort_indices = np.argsort(expand_t)
        sorted_x = expand_x[sort_indices]
        sorted_y = expand_y[sort_indices]
        sorted_t = expand_t[sort_indices]
        sorted_p = expand_p[sort_indices]
        if i < len(labels):
            samples.append((sorted_x, sorted_y, sorted_t, sorted_p))
            sample_labels.append(labels[i])
    return samples, sample_labels


def sort_key(file_path):
    parts = file_path.split(os.sep)
    x_y_part = parts[-2]  
    x, y = map(int, x_y_part.split('_'))  
    return (x, y)  

def transform(samples,labels,group_size,step_size,test):
    frames = []
    cnt = 0
    num_events = []
    for sample in samples:
        dtype = [('x', '<i8'), ('y', '<i8'), ('t', '<i8'), ('p', '<i8')]
        events_tonic = np.zeros(len(sample[0]), dtype=dtype)
        events_tonic['x'] = sample[0]*80
        events_tonic['y'] = sample[1]*60
        events_tonic['t'] = sample[2]
        events_tonic['p'] = sample[3]
        mask1 = (events_tonic['y'] >= 0) & (events_tonic['y'] < 30)
        mask2 = (events_tonic['y'] >= 30) & (events_tonic['y'] < 60)
        ratio = (mask1.sum())/(mask2.sum()+1e-6)
        num_events.append([len(sample[0]),ratio])
        # transform = tonic.transforms.ToFrame(
        #     sensor_size=(80,60,2),
        #     time_window=10000,
        # )
        # transform = tonic.transforms.ToVoxelGrid(
        #     sensor_size=(80,60,2),
        #     n_time_bins=5,
        # )
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
    return frames_groups, label_groups, num_events

def flip_events_x(events,labels,width=640,height=480):
    flip_events = events.copy()
    flip_labels = labels.copy()
    flip_events[1,:] = width-1- flip_events[1,:]
    flip_labels[:,0] = 1 - flip_labels[:,0]
    return flip_events,flip_labels

def flip_events_y(events,labels,width=640,height=480):
    flip_events = events.copy()
    flip_labels = labels.copy()
    flip_events[2,:] = height-1- flip_events[2,:]
    flip_labels[:,1] = 1 - flip_labels[:,1]
    return flip_events,flip_labels

def rotate_events(events,labels,width=640,height=480):
    rotate_events = events.copy()
    rotate_labels = labels.copy()
    angle = np.random.randint(-15,15)
    angle = angle/180*np.pi
    rotate_events[1,:] = (events[1]-width/2)*np.cos(angle) - (events[2]-height/2)*np.sin(angle) + width/2
    rotate_events[2,:] = (events[1]-width/2)*np.sin(angle) + (events[2]-height/2)*np.cos(angle) + height/2
    rotate_events = rotate_events[:,(rotate_events[1] >= 0) & (rotate_events[1] < 640) &
                              (rotate_events[2] >= 0) & (rotate_events[2] < 480)]
    
    rotate_labels[:,0] = (labels[:,0]-0.5)*np.cos(angle) - (labels[:,1]-0.5)*np.sin(angle) + 0.5
    rotate_labels[:,1] = (labels[:,0]-0.5)*np.sin(angle) + (labels[:,1]-0.5)*np.cos(angle) + 0.5
    return rotate_events,rotate_labels

def walk_and_process(root_dir,test,group_size=30,step_size=10,SLIDING=10000):
    sorted_files = []
    sorted_labels = []
    sorted_crop_labels = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                label_path = os.path.join(root, 'label.txt')
                sorted_files.append(file_path)
                sorted_labels.append(label_path)
    sorted_files.sort(key=sort_key)  
    sorted_labels.sort(key=sort_key)  
    sorted_crop_labels.sort(key=sort_key) 

    cnt = 0
    label_count = []
    for file_path, label_path in zip(sorted_files, sorted_labels):
        print(file_path, label_path)
        samples_ori, labels_ori = process_h5_and_labels(file_path, label_path,SLIDING,test)
        samples, labels = process(samples_ori,labels_ori,SLIDING)
        label_count.append(len(labels))
        print(label_count)
        frame, label, num_events= transform(samples,labels,group_size,step_size,test)
        if cnt == 0:
            frames = frame
            label_g = label
            num_events_g = num_events
        else:
            frames = np.concatenate((frames,frame),axis=0)
            label_g = np.concatenate((label_g,label),axis=0)
            num_events_g = np.concatenate((num_events_g,num_events),axis=0)
        cnt += 1
        if not test:
            samples_ori = np.array(samples_ori)
            labels_ori = np.array(labels_ori)

            event_x,label_x = flip_events_x(samples_ori,labels_ori)
            samples,labels = process(event_x,label_x,SLIDING)
            frame, label = transform(samples,labels,group_size,step_size,test)
            frames = np.concatenate((frames,frame),axis=0)
            label_g = np.concatenate((label_g,label),axis=0)

            event_y,label_y = flip_events_y(samples_ori,labels_ori)
            samples,labels = process(event_y,label_y,SLIDING)
            frame, label = transform(samples,labels,group_size,step_size,test)
            frames = np.concatenate((frames,frame),axis=0)
            label_g = np.concatenate((label_g,label),axis=0)

            event_r,label_r = rotate_events(samples_ori,labels_ori)
            samples,labels = process(event_r,label_r,SLIDING)
            frame, label = transform(samples,labels,group_size,step_size,test)
            frames = np.concatenate((frames,frame),axis=0)
            label_g = np.concatenate((label_g,label),axis=0)
    ## write num_events_g to txt file
    with open('./num_events.txt', 'w') as f:
        for item in num_events_g:
            f.write("%s\n" % item)
    return frames, label_g

group_size = 100
step_size = 100
root_dir = '/mnt/e/dataset/3ET/2025/event_data/event_data/test/'
test = True

# group_size = 100
# step_size = 50
# root_dir = '/mnt/e/dataset/3ET/2025/event_data/event_data/train/'
# test = False

EXPORT_PATH = '/mnt/e/dataset/3ET/2025/'
SLIDING = 10000
all_frame, all_labels = walk_and_process(root_dir,test,group_size,step_size,SLIDING)
print(all_frame.shape)
print(all_labels.shape)

with h5py.File(os.path.join(EXPORT_PATH,"train.h5"), 'a') as hf:
    fset = hf.create_dataset('frames', shape=all_frame.shape, maxshape = (None,group_size,2,60,80), chunks=True, dtype='float32')
    lset = hf.create_dataset('label',shape=all_labels.shape, maxshape = (None,group_size,3), chunks=True, dtype='float32')
    hf['frames'][:] = all_frame
    hf['label'][:] = all_labels