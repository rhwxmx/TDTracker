import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def p_acc(target, prediction, width_scale, height_scale, pixel_tolerances=[1, 3, 5, 10]):
    """
    Calculate the accuracy of prediction
    :param target: (N, seq_len, 2) tensor, seq_len could be 1
    :param prediction: (N, seq_len, 2) tensor
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    # target = target.cpu()
    # prediction = prediction.cpu()
    # print(target.device, prediction.device)
    target = target.reshape(-1, 2)
    prediction = prediction.reshape(-1, 2)

    dis = target - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)
    # print(dist)
    total_correct = {}
    for p_tolerance in pixel_tolerances:
        total_correct[f'p{p_tolerance}'] = torch.sum(dist < p_tolerance)

    bs_times_seqlen = target.shape[0]
    return total_correct, bs_times_seqlen


def p_acc_wo_closed_eye(target, prediction, width_scale, height_scale, pixel_tolerances=[1, 3, 5, 10]):
    """
    Calculate the accuracy of prediction, with p tolerance and only calculated on those with fully opened eyes
    :param target: (N, seqlen, 3) tensor
    :param prediction: (N, seqlen, 2) tensor, the last dimension is whether the eye is closed
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 2)
    prediction = prediction.reshape(-1, 2)

    dis = target[:, :2] - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)
    # check if there is nan in dist
    assert torch.sum(torch.isnan(dist)) == 0

    eye_closed = target[:, 2]  # 1 is closed eye
    # get the total number frames of those with fully opened eyes
    total_open_eye_frames = torch.sum(eye_closed == 0)

    # get the indices of those with closed eyes
    eye_closed_idx = torch.where(eye_closed == 1)[0]
    dist[eye_closed_idx] = np.inf
    total_correct = {}
    for p_tolerance in pixel_tolerances:
        total_correct[f'p{p_tolerance}'] = torch.sum(dist < p_tolerance)
        assert total_correct[f'p{p_tolerance}'] <= total_open_eye_frames

    return total_correct, total_open_eye_frames.item()


def px_euclidean_dist(target, prediction, width_scale, height_scale):
    """
    Calculate the total pixel euclidean distance between target and prediction
    in a batch over the sequence length
    :param target: (N, seqlen, 3) tensor
    :param prediction: (N, seqlen, 2) tensor
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    # print(target.size(),prediction.size())
    target = target.reshape(-1, 2)[:, :2]
    prediction = prediction.reshape(-1, 2)

    dis = target - prediction
    dis[:, 0] *= width_scale
    dis[:, 1] *= height_scale
    dist = torch.norm(dis, dim=-1)

    total_px_euclidean_dist = torch.sum(dist)
    sample_numbers = target.shape[0]
    return total_px_euclidean_dist, sample_numbers


class weighted_MSELoss(nn.Module):
    def __init__(self, weights, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.weights = weights
        self.mseloss = nn.MSELoss(reduction='none')
    def forward(self, inputs, targets):
        batch_loss = self.mseloss(inputs, targets) * self.weights
        if self.reduction == 'mean':
            return torch.mean(batch_loss)
        elif self.reduction == 'sum':
            return torch.sum(batch_loss)
        else:
            return batch_loss
        
class KLDiscretLoss(nn.Module):
    """
    "https://github.com/leeyegy/SimDR"
    """
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]
        self.criterion_ = nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        # print(scores.shape)
        loss = torch.mean(self.criterion_(scores, labels), dim=1)
        return loss

    def forward(self, output_x, output_y, x_p, y_p):
        num_joints = output_x.size(1)
        batch_size = output_x.size(0)
        # print(num_joints)
        loss = 0
        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx]
            coord_y_pred = output_y[:, idx]
            coord_x_gt = x_p[:, idx]*80
            coord_y_gt = y_p[:, idx]*60
            # print(coord_x_gt,coord_y_gt)
            # print(coord_x_gt,coord_y_gt)
            coord_x_gt = coord_x_gt.unsqueeze(1)
            coord_y_gt = coord_y_gt.unsqueeze(1)
            target_x = torch.zeros_like(coord_x_pred)
            target_y = torch.zeros_like(coord_y_pred)
            x = torch.arange(0, 80, 1, dtype=torch.float32).cuda()
            y = torch.arange(0, 60, 1, dtype=torch.float32).cuda()
            x = x.expand(batch_size, -1)  # Shape: (batch_size, 80)
            y = y.expand(batch_size, -1)  # Shape: (batch_size, 60)
            # print(x.shape,y.shape,coord_x_gt.shape,coord_y_gt.shape,target_x.shape,target_y.shape)
            # Now perform the operations, coord_x_gt must also be a tensor
            pi_2 = torch.tensor(torch.pi * 2, dtype=torch.float32).cuda()  # Convert to a tensor
            variance = 1.0
            target_x = (torch.exp(-((x - coord_x_gt) ** 2) / (2 * variance ** 2))) / (variance * torch.sqrt(pi_2))
            target_y = (torch.exp(-((y - coord_y_gt) ** 2) / (2 * variance ** 2))) / (variance * torch.sqrt(pi_2))
            target_x = (target_x - target_x.min()) / (target_x.max() - target_x.min())
            target_y = (target_y - target_y.min()) / (target_y.max() - target_y.min())
            loss += (self.criterion(coord_x_pred, target_x).mean())
            loss += (self.criterion(coord_y_pred, target_y).mean())
        return loss / num_joints

def decode_batch_sa_simdr(output_x, output_y):

    max_val_x, preds_x = output_x.max(2, keepdim=True)
    max_val_y, preds_y = output_y.max(2, keepdim=True)
    output_x  = F.avg_pool1d(output_x, kernel_size=10, stride=1, padding=1)
    output_y  = F.avg_pool1d(output_y, kernel_size=10, stride=1, padding=1)
    max_pro_w = torch.softmax(output_x, dim=2)
    max_pro_h = torch.softmax(output_y, dim=2)
    x_prob, max_idx = torch.max(max_pro_w, dim=-1)
    y_prob, max_idx = torch.max(max_pro_h, dim=-1)
    prob = x_prob + y_prob
    output = torch.ones([output_x.size(0), preds_x.size(1), 2])
    output[:, :, 0] = torch.squeeze(preds_x)/80
    output[:, :, 1] = torch.squeeze(preds_y)/60
    output = output.cuda()
    return output,prob



