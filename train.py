# Our code makes extensive use of code from the BTS github repository(https://github.com/cleinc/bts). We thank the authors for open sourcing their implementation.
# BTS paper -> LEE, Jin Han, et al. From big to small: Multi-scale local planar guidance for monocular depth estimation. arXiv preprint arXiv:1907.10326, 2019.

import argparse
import sys
import os
import time
import cv2
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from einops import rearrange, reduce, repeat

from tensorboardX import SummaryWriter
from tqdm import tqdm

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

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='kd_ude')
parser.add_argument('--teacher_gpu',                type=int,   help='gpu num for teacher network', default=0)
parser.add_argument('--student_gpu',                type=int,   help='gpu num for student network', default=1)

# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--data_teacher_path',         type=str,   help='path to the data', required=True)
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=352)
parser.add_argument('--input_width',               type=int,   help='input width',  default=704)
parser.add_argument('--min_depth',                 type=float, help='minimum depth in estimation', default=1e-3)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=80)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=500)

# Training
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Online eval
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                    'if empty outputs to checkpoint folder', default='')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.mode == 'train' and args.checkpoint_path:
    model_dir = os.path.dirname(args.checkpoint_path)
    model_name = os.path.basename(model_dir)
    import sys
    sys.path.append(model_dir)
    for key, val in vars(__import__(model_name)).items():
        if key.startswith('__') and key.endswith('__'):
            continue
        vars()[key] = val

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

class silog_loss_multiscale(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss_multiscale, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, low1, low2, low3, depth_gt, mask):
        d1 = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        loss1 = torch.sqrt((d1 ** 2).mean() - self.variance_focus * (d1.mean() ** 2)) * 10.0
        d2 = torch.log(low1[mask]) - torch.log(depth_gt[mask])
        loss2 = torch.sqrt((d2 ** 2).mean() - self.variance_focus * (d2.mean() ** 2)) * 10.0
        d3 = torch.log(low2[mask]) - torch.log(depth_gt[mask])
        loss3 = torch.sqrt((d3 ** 2).mean() - self.variance_focus * (d3.mean() ** 2)) * 10.0
        d4 = torch.log(low3[mask]) - torch.log(depth_gt[mask])
        loss4 = torch.sqrt((d4 ** 2).mean() - self.variance_focus * (d4.mean() ** 2)) * 10.0
        return loss1 + ((1//4) * loss2) + ((1//16) * loss3) + ((1//64) * loss4)

def median_scaling(pred, gt, args):
    _,_,gt_height,gt_width = gt.shape
    
    pred_depth = pred.cpu().detach().numpy()
    gt_depth = gt.cpu().detach().numpy()
    
    if args.dataset == "kitti":
        mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[:,:,crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)
    else:
        mask = gt_depth > 0
        
    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]
    
    scaling_factor = np.median(gt_depth) / np.median(pred_depth)
    
    return scaling_factor

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)

def online_eval(student_model, dataloader_eval, gpu, args):
    eval_measures = torch.zeros(10).cuda(device=gpu)
    for idx, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            focal = torch.autograd.Variable(eval_sample_batched['focal'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue
            gt_depth = rearrange(gt_depth, 'b h w c -> b c h w')
            _, _, _, pred_depth = student_model(image, focal)
            
            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval
        
        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

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
        
        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples'.format(int(cnt)))
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms','sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    for i in range(8):
        print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.3f}'.format(eval_measures_cpu[8]))
    return eval_measures_cpu

def trainer(args, device_teacher, device_student):

    print("Use GPU for teacher network: {} for training".format(device_teacher))
    print("Use GPU for student network: {} for training".format(device_student))
    
    # Create student network
    teacher_model = DPTDepthModel_T(
            path='./models/DPT_teacher/teacher_depth.pth',
            scale=0.00006016,
            shift=0.00579,
            invert=False,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
    teacher_model.eval()
    teacher_model.to(device_teacher)
    
    # Create student network
    student_model = DPTDepthModel_S(
            args, path='./models/DPT_student/dpt_large-midas-2f21e586.pt',
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
    student_model.train()
    student_model.to(device_student)

    num_params_student = sum([np.prod(p.size()) for p in student_model.parameters()])
    print("Total number of parameters in Student network: {}".format(num_params_student))
    num_params_update_student = sum([np.prod(p.shape) for p in student_model.parameters() if p.requires_grad])
    print("Total number of learning parameters in Student network: {}".format(num_params_update_student))
    # Optimizer for trainable parameters in student network
    optimizer = torch.optim.AdamW([{'params': student_model.pretrained.parameters(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay}, {'params': student_model.scratch.parameters(), 'lr': (args.learning_rate*10), 'weight_decay': 0}], eps=args.adam_eps)

    # Declare dataloader
    dataloader = CustomDataLoader(args, 'train')
    dataloader_eval = CustomDataLoader(args, 'online_eval')

    # Setting for training monitoring
    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)

    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            global_step = checkpoint['global_step']
            student_model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                best_eval_steps = checkpoint['best_eval_steps']
            except KeyError:
                print("Could not load values for online evaluation")

            print("Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True

    writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
    if args.do_online_eval:
        if args.eval_summary_directory != '':
            eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
        else:
            eval_summary_path = os.path.join(args.log_directory, 'eval')
        eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)
    
    silog_criterion_multi = silog_loss_multiscale(variance_focus=args.variance_focus)

    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    var_sum = [var.sum().cpu().detach().numpy() for var in student_model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    # Train process
    while epoch < args.num_epochs:
        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()

            image_student = torch.autograd.Variable(sample_batched['image'].cuda(device_student, non_blocking=True))
            focal_student = torch.autograd.Variable(sample_batched['focal'].cuda(device_student, non_blocking=True))
            
            image_teacher = torch.autograd.Variable(sample_batched['image_teacher'].cuda(device_teacher, non_blocking=True))
            
            _,_,gt_height, gt_width = image_teacher.shape
            image_teacher = nn.functional.interpolate(image_teacher, size=(320,1024), mode='bilinear',align_corners=True)
            
            _, pred = teacher_model(image_teacher)
            pseudo_gt, _ = disp_to_depth(pred[("disp", 0)], args.min_depth*100, args.max_depth+20)
            pseudo_gt = nn.functional.interpolate(pseudo_gt, size=(gt_height,gt_width), mode='bilinear',align_corners=True)
            pseudo_gt = 1 / pseudo_gt
            pseudo_gt = 5.4  * pseudo_gt.to(device_student)
            
            low3, low2, low1, depth_est = student_model(image_student, focal_student)

            if args.dataset == 'nyu':
                mask = pseudo_gt > 0.1
            else:
                mask = pseudo_gt > 1.0
            
            # distillation loss
            #loss = silog_criterion.forward(depth_est, pseudo_gt, mask.to(torch.bool))
            loss = silog_criterion_multi.forward(depth_est, low1, low2, low3, pseudo_gt, mask.to(torch.bool))
            
            loss.backward()
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr
            optimizer.step()

            print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
            if np.isnan(loss.cpu().item()):
                print('NaN in loss occurred. Aborting training.')
                return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum = [var.sum().cpu().detach().numpy() for var in student_model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar

                print("{}".format(args.model_name))
                print_string = '| examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))

                writer.add_scalar('silog_loss', loss, global_step)
                writer.add_scalar('learning_rate', current_lr, global_step)
                writer.add_scalar('var average', var_sum.item()/var_cnt, global_step)
                pseudo_gt = torch.where(pseudo_gt < 1e-3, pseudo_gt * 0 + 1e3, pseudo_gt)
                for i in range(num_log_images):
                    writer.add_image('pseudo_gt/image/{}'.format(i),
                                        normalize_result(1 / pseudo_gt[i, :, :, :].data), global_step)
                    writer.add_image('depth_est/image/{}'.format(i),
                                        normalize_result(1 / depth_est[i, :, :, :].data), global_step)
                    writer.add_image('image/image/{}'.format(i), inv_normalize(image_student[i, :, :, :]).data,global_step)
                writer.flush()

            if not args.do_online_eval and global_step and global_step % args.save_freq == 0:
                checkpoint = {'global_step': global_step,
                            'model': student_model.state_dict(),
                            'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, args.log_directory + '/' + args.model_name + '/model-{}'.format(global_step))

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                time.sleep(0.1)
                student_model.eval()
                eval_measures = online_eval(student_model, dataloader_eval, device_student, args)
                if eval_measures is not None:
                    for i in range(9):
                        eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item()
                            best_eval_measures_higher_better[i-6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                            model_path = args.log_directory + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                            print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                            checkpoint = {'global_step': global_step,
                                          'model': student_model.state_dict(),
                                          'optimizer': optimizer.state_dict(),
                                          'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                          'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                          'best_eval_steps': best_eval_steps
                                          }
                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                    eval_summary_writer.flush()
                student_model.train()
                block_print()
                enable_print()
            global_step += 1
        epoch += 1
    if args.do_online_eval:
        eval_summary_writer.close()

def main():
    device_teacher = torch.device(f'cuda:{args.teacher_gpu}' if torch.cuda.is_available() else 'cpu')
    device_student = torch.device(f'cuda:{args.student_gpu}' if torch.cuda.is_available() else 'cpu')
    trainer(args, device_teacher, device_student)

if __name__ == '__main__':
    main()