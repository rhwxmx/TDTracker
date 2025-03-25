import pandas as pd
from metrics import *
from scipy.ndimage import binary_dilation
import csv
pixel_tolerances = [1,3,5,10]
df = pd.read_csv('./submission.csv')

predict_x,predict_y = df['x'].values,df['y'].values
target_x,target_y = df['x_gt'].values,df['y_gt'].values
label = np.stack([target_x,target_y],axis=1)
outputs = np.stack([predict_x,predict_y],axis=1)
label = torch.from_numpy(label)
outputs = torch.from_numpy(outputs)

### Remove samples with low confidence and replace them ###
ratio  = df['prob'].values
mask = ratio < 0.5
indices = np.where(mask)[0]
for j in indices:
    i = j.copy()
    while i in indices:
        i -= 1
    outputs[j] = outputs[i]

### Remove samples with eye closure and replace them ###
ratio  = df['num_events_y'].values
mask = ratio < 0.09
mask = binary_dilation(mask,iterations=2)
indices = np.where(mask)[0]
for j in indices:
    i = j.copy()
    while i in indices:
        i -= 1
    outputs[j] = outputs[i]

outputs = outputs.numpy()
csv_file = './submission_postprocess.csv'
numbers = []
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['row_id', 'x', 'y'])
    for i in range(len(outputs)):
        x, y = outputs[i][0], outputs[i][1]
        writer.writerow([i, x, y])

p_error_total, bs_times_seqlen = px_euclidean_dist(label, outputs,
                                                    width_scale=1,
                                                    height_scale=1)
print("final_mse_error",p_error_total/bs_times_seqlen)