# Copyright 2021 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm

from utils.vis import save_debug_3d_images, save_debug_3d_json, save_demo_3d_json

from models.util.misc import get_total_grad_norm, is_main_process
from easymocap.mytools.vis_base import plot_keypoints_auto

IMAGE_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
IMAGE_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


logger = logging.getLogger(__name__)


def train_3d(config, model, optimizer, loader, epoch, output_dir, writer_dict,
             device=torch.device('cuda'), num_views=5, finetune_backbone=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_ce = AverageMeter()
    class_error = AverageMeter()
    loss_pose_perjoint = AverageMeter()
    loss_pose_perbone = AverageMeter()
    loss_pose_perprojection = AverageMeter()
    cardinality_error = AverageMeter()

    model.train()

    if model.module.backbone is not None:
        # Comment out this line if you want to train 2D backbone jointly
        if not finetune_backbone:
            model.module.backbone.eval()

    threshold = model.module.pred_conf_threshold

    end = time.time()
    for i, (inputs, meta) in tqdm(enumerate(loader)):
        assert len(inputs) == num_views
        inputs = [i.to(device) for i in inputs]
        meta = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()} for t in meta]
        data_time.update(time_synchronized() - end)
        end = time_synchronized()

        out, loss_dict = model(views=inputs, meta=meta)

        gt_3d = meta[0]['joints_3d'].float()
        num_joints = gt_3d.shape[2]
        bs, num_queries = out["pred_logits"].shape[:2]

        src_poses = out['pred_poses']['outputs_coord'].\
            view(bs, num_queries, num_joints, 3)
        src_poses = model.module.norm2absolute(src_poses)
        score = out['pred_logits'][:, :, 1:2].sigmoid()
        score = score.unsqueeze(2).expand(-1, -1, num_joints, -1)
        temp = (score > threshold).float() - 1

        pred = torch.cat([src_poses, temp, score], dim=-1)

        weight_dict = model.module.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)

        loss_ce.update(loss_dict['loss_ce'].sum().item())
        class_error.update(loss_dict['class_error'].sum().item())

        loss_pose_perjoint.update(loss_dict['loss_pose_perjoint'].sum().item())
        if 'loss_pose_perbone' in loss_dict:
            loss_pose_perbone.update(
                loss_dict['loss_pose_perbone'].sum().item())
        if 'loss_pose_perprojection' in loss_dict:
            loss_pose_perprojection.update(
                loss_dict['loss_pose_perprojection'].sum().item())

        cardinality_error.update(
            loss_dict['cardinality_error'].sum().item())

        if losses > 0:
            optimizer.zero_grad()
            losses.backward()
            if config.TRAIN.clip_max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.clip_max_norm)
            else:
                grad_total_norm = get_total_grad_norm(
                    model.parameters(), config.TRAIN.clip_max_norm)

            optimizer.step()

        batch_time.update(time_synchronized() - end)
        end = time_synchronized()

        if i % config.PRINT_FREQ == 0 and is_main_process():
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = \
                'Epoch: [{0}][{1}/{2}]\t' \
                'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                'Speed: {speed:.1f} samples/s\t' \
                'Data: {data_time.val:.3f}s ' '({data_time.avg:.3f}s)\t' \
                'loss_ce: {loss_ce.val:.7f} ' '({loss_ce.avg:.7f})\t' \
                'class_error: {class_error.val:.7f} ' \
                '({class_error.avg:.7f})\t' \
                'loss_pose_perjoint: {loss_pose_perjoint.val:.6f} ' \
                '({loss_pose_perjoint.avg:.6f})\t' \
                'loss_pose_perbone: {loss_pose_perbone.val:.6f} ' \
                '({loss_pose_perbone.avg:.6f})\t' \
                'loss_pose_perprojection: {loss_pose_perprojection.val:.6f} ' \
                '({loss_pose_perprojection.avg:.6f})\t' \
                'cardinality_error: {cardinality_error.val:.6f} ' \
                '({cardinality_error.avg:.6f})\t' \
                'Memory {memory:.1f}\t'\
                'gradnorm {gradnorm:.2f}'.format(
                  epoch, i, len(loader),
                  batch_time=batch_time,
                  speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                  data_time=data_time,
                  loss_ce=loss_ce,
                  class_error=class_error,
                  loss_pose_perjoint=loss_pose_perjoint,
                  loss_pose_perbone=loss_pose_perbone,
                  loss_pose_perprojection=loss_pose_perprojection,
                  cardinality_error=cardinality_error,
                  memory=gpu_memory_usage,
                  gradnorm=grad_total_norm)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.item(), global_steps)
            writer.add_scalar('train_loss_ce', loss_ce.val, global_steps)
            writer.add_scalar('train_loss_pose_perbone', loss_pose_perbone.val, global_steps)
            writer.add_scalar('train_loss_pose_perjoint', loss_pose_perjoint.val, global_steps)
            writer.add_scalar('train_loss_pose_perprojection', loss_pose_perprojection.val, global_steps)
            writer.add_scalar('train_class_error', class_error.val, global_steps)
            writer.add_scalar('train_card_error', cardinality_error.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            subs = meta[0]['key'][0].split('_')
            prefix2 = '{}/{:06}_{}_{}_{}'.format(
                os.path.join(output_dir, 'train'), global_steps, subs[0], subs[1], subs[-1])
            save_debug_3d_images(config, meta[0], pred, prefix2)


def validate_3d(config, model, loader, output_dir, threshold, num_views=5, epoch=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()

    preds = []
    meta_image_files = []
    with torch.no_grad():
        end = time.time()
        for i, (inputs, meta) in tqdm(enumerate(loader)):
            data_time.update(time.time() - end)
            assert len(inputs) == num_views

            output = model(views=inputs, meta=meta)

            meta_image_files.append(meta[0]['image'])
            gt_3d = meta[0]['joints_3d'].float()
            num_joints = gt_3d.shape[2]
            bs, num_queries = output["pred_logits"].shape[:2]

            # inter_poses_xy = output['inter_poses_xy'].view(-1, bs, 5, num_queries, num_joints, 2)
            # raw_poses = output['raw_poses'].view(bs, num_queries, num_joints, 3)
            # raw_poses = model.module.norm2absolute(raw_poses)
            # inter_poses = output['inter_poses'].view(-1, bs, num_queries, num_joints, 3)
            # init_poses = output['init_poses']['outputs_coord'].\
            #     view(bs, num_queries, num_joints, 3)
            # init_poses = model.module.norm2absolute(init_poses)
            src_poses = output['pred_poses']['outputs_coord'].\
                view(bs, num_queries, num_joints, 3)
            src_poses = model.module.norm2absolute(src_poses)
            score = output['pred_logits'][:, :, 1:2].sigmoid()
            score = score.unsqueeze(2).expand(-1, -1, num_joints, -1)
            temp = (score > threshold).float() - 1

            # raw = torch.cat([raw_poses, torch.zeros_like(temp), score], dim=-1)
            # # raw = torch.cat([raw_poses, temp, score], dim=-1)
            # raw = raw.detach().cpu().numpy()
            # inter_poses_list = []
            # os.makedirs(os.path.join(output_dir, '2dkp'), exist_ok=True)
            # for i in range(inter_poses.shape[0]):
            #     inter_pose = torch.cat([inter_poses[i], torch.zeros_like(temp), score], dim=-1)
            #     inter_pose = inter_pose.detach().cpu().numpy()
            #     inter_poses_list.append(inter_pose)
            #     inter_pose_xy = inter_poses_xy[i][0].detach().cpu().numpy()
            #     for j in range(5):
            #         img = inputs[j][0].cpu().numpy()
            #         img = img.transpose(1, 2, 0)
            #         img = img[..., [2, 1, 0]]
            #         img = (img * IMAGE_STD + IMAGE_MEAN) * 255
            #         img = np.ascontiguousarray(img).astype('uint8')
            #         for k in range(num_queries):
            #             plot_keypoints_auto(img, inter_pose_xy[j, k], pid=k, config_name='panoptic15')
            #         cv2.imwrite(os.path.join(output_dir, '2dkp', f'{j:0>2}.jpg'), img)
            # # init = torch.cat([init_poses, torch.zeros_like(temp), score], dim=-1)
            # init = torch.cat([init_poses, temp, score], dim=-1)
            # init = init.detach().cpu().numpy()
            pred = torch.cat([src_poses, temp, score], dim=-1)
            pred = pred.detach().cpu().numpy()
            for b in range(pred.shape[0]):
                preds.append(pred[b])

            # batch_time.update(time.time() - end)
            # end = time.time()
            # if (i % config.PRINT_FREQ == 0 or i == len(loader) - 1) \
            #         and is_main_process():
            #     gpu_memory_usage = torch.cuda.memory_allocated(0)
            #     msg = 'Test: [{0}/{1}]\t' \
            #           'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
            #           'Speed: {speed:.1f} samples/s\t' \
            #           'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
            #           'Memory {memory:.1f}'.format(
            #             i, len(loader), batch_time=batch_time,
            #             speed=len(inputs) * inputs[0].size(0) / batch_time.val,
            #             data_time=data_time, memory=gpu_memory_usage)
            #     logger.info(msg)
            #     subs = meta[0]['key'][0].split('_')
            #     if epoch is None:
            #         prefix2 = '{}/val_{}_{}_{}'.format(
            #             os.path.join(output_dir, 'validation'), subs[0], subs[1], subs[-1])
            #     else:
            #         prefix2 = '{}/{:03}_{}_{}_{}'.format(
            #             os.path.join(output_dir, 'validation'), epoch, subs[0], subs[1], subs[-1])
            #     save_debug_3d_images(config, meta[0], pred, prefix2)

            # if epoch is not None:
            #     pass
            # else:
            if is_main_process():
                # key = meta[0]['key'][0].split('_')
                # if 'Hug' in key[1]:
                #     save_debug_3d_json(config, meta[0], pred, output_dir, vis=True)
                # else:
                save_debug_3d_json(config, meta[0], pred, output_dir, vis=False)
                # save_debug_3d_json(config, meta[0], pred, os.path.join(output_dir, 'debug_pred'), vis=True)
                # save_debug_3d_json(config, meta[0], init, os.path.join(output_dir, 'debug_init'), vis=True)
                # for i in range(len(inter_poses_list)):
                    # save_debug_3d_json(config, meta[0], inter_poses_list[i], os.path.join(output_dir, f'debug_inter_{i}'), vis=True)

    return preds, meta_image_files

def demo_3d(config, model, loader, output_dir, threshold, num_views=5, epoch=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()

    preds = []
    meta_image_files = []
    with torch.no_grad():
        end = time.time()
        for i, (inputs, meta) in tqdm(enumerate(loader)):
            data_time.update(time.time() - end)
            assert len(inputs) == num_views

            output = model(views=inputs, meta=meta)

            meta_image_files.append(meta[0]['image'])
            gt_3d = meta[0]['joints_3d'].float()
            num_joints = gt_3d.shape[2]
            bs, num_queries = output["pred_logits"].shape[:2]

            # inter_poses_xy = output['inter_poses_xy'].view(-1, bs, 5, num_queries, num_joints, 2)
            # raw_poses = output['raw_poses'].view(bs, num_queries, num_joints, 3)
            # raw_poses = model.module.norm2absolute(raw_poses)
            # inter_poses = output['inter_poses'].view(-1, bs, num_queries, num_joints, 3)
            # init_poses = output['init_poses']['outputs_coord'].\
            #     view(bs, num_queries, num_joints, 3)
            # init_poses = model.module.norm2absolute(init_poses)
            src_poses = output['pred_poses']['outputs_coord'].\
                view(bs, num_queries, num_joints, 3)
            src_poses = model.module.norm2absolute(src_poses)
            score = output['pred_logits'][:, :, 1:2].sigmoid()
            score = score.unsqueeze(2).expand(-1, -1, num_joints, -1)
            temp = (score > threshold).float() - 1

            # raw = torch.cat([raw_poses, torch.zeros_like(temp), score], dim=-1)
            # # raw = torch.cat([raw_poses, temp, score], dim=-1)
            # raw = raw.detach().cpu().numpy()
            # inter_poses_list = []
            # os.makedirs(os.path.join(output_dir, '2dkp'), exist_ok=True)
            # for i in range(inter_poses.shape[0]):
            #     inter_pose = torch.cat([inter_poses[i], torch.zeros_like(temp), score], dim=-1)
            #     inter_pose = inter_pose.detach().cpu().numpy()
            #     inter_poses_list.append(inter_pose)
            #     inter_pose_xy = inter_poses_xy[i][0].detach().cpu().numpy()
            #     for j in range(5):
            #         img = inputs[j][0].cpu().numpy()
            #         img = img.transpose(1, 2, 0)
            #         img = img[..., [2, 1, 0]]
            #         img = (img * IMAGE_STD + IMAGE_MEAN) * 255
            #         img = np.ascontiguousarray(img).astype('uint8')
            #         for k in range(num_queries):
            #             plot_keypoints_auto(img, inter_pose_xy[j, k], pid=k, config_name='panoptic15')
            #         cv2.imwrite(os.path.join(output_dir, '2dkp', f'{j:0>2}.jpg'), img)
            # # init = torch.cat([init_poses, torch.zeros_like(temp), score], dim=-1)
            # init = torch.cat([init_poses, temp, score], dim=-1)
            # init = init.detach().cpu().numpy()
            pred = torch.cat([src_poses, temp, score], dim=-1)
            pred = pred.detach().cpu().numpy()
            for b in range(pred.shape[0]):
                preds.append(pred[b])

            # batch_time.update(time.time() - end)
            # end = time.time()
            # if (i % config.PRINT_FREQ == 0 or i == len(loader) - 1) \
            #         and is_main_process():
            #     gpu_memory_usage = torch.cuda.memory_allocated(0)
            #     msg = 'Test: [{0}/{1}]\t' \
            #           'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
            #           'Speed: {speed:.1f} samples/s\t' \
            #           'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
            #           'Memory {memory:.1f}'.format(
            #             i, len(loader), batch_time=batch_time,
            #             speed=len(inputs) * inputs[0].size(0) / batch_time.val,
            #             data_time=data_time, memory=gpu_memory_usage)
            #     logger.info(msg)
            #     subs = meta[0]['key'][0].split('_')
            #     if epoch is None:
            #         prefix2 = '{}/val_{}_{}_{}'.format(
            #             os.path.join(output_dir, 'validation'), subs[0], subs[1], subs[-1])
            #     else:
            #         prefix2 = '{}/{:03}_{}_{}_{}'.format(
            #             os.path.join(output_dir, 'validation'), epoch, subs[0], subs[1], subs[-1])
            #     save_debug_3d_images(config, meta[0], pred, prefix2)

            # if epoch is not None:
            #     pass
            # else:
            if is_main_process():
                # key = meta[0]['key'][0].split('_')
                # if 'Hug' in key[1]:
                #     save_debug_3d_json(config, meta[0], pred, output_dir, vis=True)
                # else:
                save_demo_3d_json(config, meta[0], pred, output_dir, vis=False)
                # save_debug_3d_json(config, meta[0], pred, os.path.join(output_dir, 'debug_pred'), vis=True)
                # save_debug_3d_json(config, meta[0], init, os.path.join(output_dir, 'debug_init'), vis=True)
                # for i in range(len(inter_poses_list)):
                    # save_debug_3d_json(config, meta[0], inter_poses_list[i], os.path.join(output_dir, f'debug_inter_{i}'), vis=True)

    return preds, meta_image_files

def speed_3d(config, model, loader, output_dir, threshold, num_views=5, epoch=None):
    full_time_metric = AverageMeter()
    pose_time_metric = AverageMeter()
    model.eval()

    preds = []
    meta_image_files = []
    with torch.no_grad():
        for i, (inputs, meta) in enumerate(loader):
            assert len(inputs) == num_views

            output, full_time, pose_time = model(views=inputs, meta=meta, test_time=True)

            meta_image_files.append(meta[0]['image'])
            gt_3d = meta[0]['joints_3d'].float()
            num_joints = gt_3d.shape[2]
            bs, num_queries = output["pred_logits"].shape[:2]

            src_poses = output['pred_poses']['outputs_coord'].\
                view(bs, num_queries, num_joints, 3)
            src_poses = model.module.norm2absolute(src_poses)
            score = output['pred_logits'][:, :, 1:2].sigmoid()
            score = score.unsqueeze(2).expand(-1, -1, num_joints, -1)
            temp = (score > threshold).float() - 1

            pred = torch.cat([src_poses, temp, score], dim=-1)
            pred = pred.detach().cpu().numpy()
            for b in range(pred.shape[0]):
                preds.append(pred[b])
            
            full_time_metric.update(full_time)
            pose_time_metric.update(pose_time)

            print(full_time_metric.avg)
            print(pose_time_metric.avg)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
