import os
import sys
import numpy as np
import h5py
import open3d as o3d
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

def load_h5(h5_filename):
    print(h5_filename)
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    frame = f['frames'][:]
    # print(frame,data,label)
    return frame, data, label

def load_h5_f(h5_filename):
    print(h5_filename)
    f = h5py.File(h5_filename)
    label = f['label'][:]
    frame = f['frames'][:]
    # print(frame,data,label)
    return frame, label

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def loadDataFile(filename):
    return load_h5(filename)

