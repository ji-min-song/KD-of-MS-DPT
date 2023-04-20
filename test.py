# Our code makes extensive use of code from the BTS github repository(https://github.com/cleinc/bts). We thank the authors for open sourcing their implementation.
# BTS paper -> LEE, Jin Han, et al. From big to small: Multi-scale local planar guidance for monocular depth estimation. arXiv preprint arXiv:1907.10326, 2019.

from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

import errno
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from models.DPT_teacher.models import DPTDepthModel as DPTDepthModel_T
from models.DPT_student.models import DPTDepthModel as DPTDepthModel_S
from models.Monodepth2.depth_network import *
from custom_dataloader import *
from layers import *

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description='KD_UDE PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name',                type=str,   help='model name', default='kd_ude')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--gt_path',             type=str,   help='root path to the groundtruth data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='kitti')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--gt_version',                               help='choose gt_version between improved or original', default='improved')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)
    
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3

def test(params):
    """Test function."""
    device = torch.device("cuda")
    args = params
    args.mode = 'test'
    dataloader = CustomDataLoader(args, 'test')
    model = DPTDepthModel_S(
            args, path='./models/DPT_student/dpt_large-midas-2f21e586.pt',
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint_path, map_location="cuda:0")
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_test_samples = len(dataloader.data)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))
    
    missing_ids = []
    gt_depths = []
    pred_depths = []
    
    start_time = time.time()
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(dataloader.data)):
            gt_depth_path = os.path.join(args.gt_path, lines[idx].split()[1])
            if args.gt_version == 'original':
                gt_depth_path.replace('groundtruth','velodyne_raw')
            gt_depth = cv2.imread(gt_depth_path, -1)
            if gt_depth is None:
                missing_ids.append(idx)
                continue
            gt_depth = gt_depth.astype(np.float32) / 256.0
            gt_depths.append(gt_depth)
                
            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())
            
            _, _, _, depth_est = model(image, focal)
            
            pred_depths.append(depth_est)
    
    print(f'Number of total samples: {len(dataloader.data)}, Number of missing samples: {len(missing_ids)}, Number of evaluation samples: {len(dataloader.data)-len(missing_ids)}')
    assert len(gt_depths) == len(pred_depths)
    
    silog = np.zeros(len(gt_depths), np.float32)
    log10 = np.zeros(len(gt_depths), np.float32)
    rms = np.zeros(len(gt_depths), np.float32)
    log_rms = np.zeros(len(gt_depths), np.float32)
    abs_rel = np.zeros(len(gt_depths), np.float32)
    sq_rel = np.zeros(len(gt_depths), np.float32)
    d1 = np.zeros(len(gt_depths), np.float32)
    d2 = np.zeros(len(gt_depths), np.float32)
    d3 = np.zeros(len(gt_depths), np.float32)
    
    for i, gt_depth in enumerate(gt_depths):
        pred_depth = pred_depths[i].cpu().detach().numpy()
        
        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        gt_depth[np.isinf(gt_depth)] = 0
        gt_depth[np.isnan(gt_depth)] = 0

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                else:
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)
        
        silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('d1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10'))
    print("{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format( d1.mean(), d2.mean(), d3.mean(), abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), silog.mean(), log10.mean()))
    
    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')
    
    return silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3

if __name__ == '__main__':
    test(args)
