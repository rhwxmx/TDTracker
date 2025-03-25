import os
import sys
import datetime
import logging
import argparse
import provider_data
from pathlib import Path
from metrics import *
from tqdm import tqdm
import csv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import time
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--test_h5_path', type=str, default='E://dataset//3ET//2025//test_200_200.h5', help='test_data')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--sensor_width', type=int, default=640, help='sensor width')
    parser.add_argument('--sensor_height', type=int, default=480, help='sensor height')
    parser.add_argument('--spatial_factor', type=float, default=0.125, help='spatial factor')
    parser.add_argument('--pixel_tolerances', default=[1, 3, 5, 10, 15], type=int,  help='pixel_tolerances')
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def validate(net, val_loader,test=1):
    net.eval()
    total_p_corr_all = {f'p{p}_all': 0 for p in args.pixel_tolerances}
    total_p_error_all = {f'error_all': 0}
    total_samples_all, total_sample_p_error_all = 0, 0
    if test:
        label_num = [699,565,672,412,669,905,1039,736,1160,645,1238]
        label_file = ['1_1','2_2','3_1','4_2','5_2','6_4','7_5','8_2','8_3','10_2','12_4']
    else:
        label_num = [429, 761, 976, 520, 989, 665, 1029, 1215, 460, 577, 652, 890, 989, 549, 873, 700, 738, 690, 864, 660, 874, 1146, 990, 1541, 1089, 1202, 623, 979, 751, 711, 551, 925, 1053, 591, 837, 1072, 519, 1015, 926, 936]
        label_file = ['1_2','1_3','1_4','1_5','1_6','2_1','2_3','2_4','2_5','3_2','4_1','4_3','4_4','4_5','5_1','6_2','6_3','6_5','7_1','7_2','7_3','7_4','8_1','8_4','8_5','9_1','9_2','9_3','9_4','9_5','10_1','10_3','11_1','11_2','11_4','12_1','12_2','12_3','12_5','13_1']
    cnt = 0
    index =0
    numbatch = 200
    drop = numbatch
    sample_count = 0
    frame_label = []
    gt_label = []
    probs = []
    with torch.no_grad():
        for (frame, label) in tqdm(val_loader, desc="Processing validation data"):
            cnt += 1
            if cnt*numbatch > label_num[index]:
                drop = label_num[index] - (cnt-1)*numbatch
                index+=1
                cnt = 0
                print(cnt, drop, index)
            sample_count += drop
            label = label.float().cuda()
            predict_w,predict_h = net(frame.float().cuda())
            b,s,c,h,w = frame.shape
            outputs,prob = decode_batch_sa_simdr(predict_w,predict_h)
            frame = frame.squeeze(0)
            p_corr, batch_size = p_acc(label[:, :drop,:2], outputs[:, :drop, :2],
                                       width_scale=args.sensor_width * args.spatial_factor,
                                       height_scale=args.sensor_height * args.spatial_factor,
                                       pixel_tolerances=args.pixel_tolerances)
            total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in
                                args.pixel_tolerances}
            total_samples_all += batch_size
            p_error_total, bs_times_seqlen = px_euclidean_dist(label[:, :drop,:2], outputs[:, :drop, :2],
                                                               width_scale=args.sensor_width * args.spatial_factor,
                                                               height_scale=args.sensor_height * args.spatial_factor)
            total_p_error_all = {f'error_all': (total_p_error_all[f'error_all'] + p_error_total).item()}
            total_sample_p_error_all += bs_times_seqlen
            
            frame_label.append(outputs[:, :drop, :2].reshape(-1,2).cpu().numpy())
            gt_label.append(label[:, :drop, :2].reshape(-1,2).cpu().numpy())
            probs.append(prob[:,:drop].cpu().numpy())
            drop = numbatch
        frame_label = np.concatenate(frame_label, axis=0)
        frame_label = np.hstack([frame_label[:, :2] * [80, 60], np.zeros((frame_label.shape[0], 1))])
        gt_label = np.concatenate(gt_label, axis=0)
        gt_label = np.hstack([gt_label[:, :2] * [80, 60], np.zeros((gt_label.shape[0], 1))])
        probs = np.concatenate(probs, axis=1)

    if test:
        csv_file = './submission.csv'
        numbers = []
        with open('./num_events.txt', 'r') as f:
            num_events_g = f.readlines()
            for item in num_events_g:
                cleaned_item = item.replace('[', '').replace(']', '') 
                number = [float(x) for x in cleaned_item.split()]
                numbers.append(number)      
        
        print(len(frame_label))
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['row_id', 'x', 'y', 'x_gt', 'y_gt', 'num_events_x', 'num_events_y','prob'])
            # writer.writerow(['row_id', 'x', 'y'])
            for i in range(len(frame_label)):
                x, y = frame_label[i][0], frame_label[i][1]
                writer.writerow([i, x, y, gt_label[i][0], gt_label[i][1],numbers[i][0],numbers[i][1],probs[0][i]])
                # writer.writerow([i, x, y])
    metrics = {'val_p_acc_all': {f'val_p{k}_acc_all': (total_p_corr_all[f'p{k}_all'] / total_samples_all) for k in
                                 args.pixel_tolerances},
               'val_p_error_all': {f'val_p_error_all': (total_p_error_all[f'error_all'] / total_sample_p_error_all)}}
    return metrics

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/pointmlp.txt' % (log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    F_test, L_test = provider_data.load_h5_f(args.test_h5_path)
    F_test = np.array(F_test)
    L_test = np.array(L_test)
    F_test = torch.from_numpy(F_test)
    L_test = torch.from_numpy(L_test)
    print('data=', F_test.shape)
    dataset = torch.utils.data.TensorDataset(F_test, L_test)
    valDataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    from models.TDTracker import Model
    classifier = Model(args)
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
    checkpoint = torch.load('./best_checkpoint.pth')
    classifier.load_state_dict(checkpoint,strict=False)
    log_string('Use pretrain model')
    net = classifier.eval()
    val_metrics = validate(net, valDataLoader)
    print('val_acc=', val_metrics['val_p_acc_all'])
    print('val_error=', val_metrics['val_p_error_all'])
    logger.info('End of training...')
if __name__ == '__main__':
    args = parse_args()
    main(args)
