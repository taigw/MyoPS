# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import time
import scipy
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from scipy import special
from datetime import datetime
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from pymic.io_my.image_read_write import save_nd_array_as_image
from pymic.io_my.nifty_dataset import NiftyDatasetContour
from pymic.io_my.transform3d import get_transform
from pymic.net_run.net_run import TrainInferAgent
from pymic.net_run.infer_func import volume_infer
from pymic.net_run.loss import *
from pymic.net_run.get_optimizer import get_optimiser
from pymic.util.image_process import convert_label
from pymic.util.parse_config import parse_config


class TrainInferAgentContour(TrainInferAgent):

    def get_stage_dataset_from_config(self, stage):
        assert(stage in ['train', 'valid', 'test'])
        root_dir  = self.config['dataset']['root_dir']
        modal_num = self.config['dataset']['modal_num']
        if(stage == "train" or stage == "valid"):
            transform_names = self.config['dataset']['train_transform']
        elif(stage == "test"):
            transform_names = self.config['dataset']['test_transform']
        else:
            raise ValueError("Incorrect value for stage: {0:}".format(stage))

        self.transform_list = [get_transform(name, self.config['dataset']) \
                    for name in transform_names ]    
        csv_file = self.config['dataset'].get(stage + '_csv', None)
        dataset  = NiftyDatasetContour(root_dir=root_dir,
                                csv_file  = csv_file,
                                modal_num = modal_num,
                                with_label= not (stage == 'test'),
                                transform = transforms.Compose(self.transform_list))
        return dataset

    def train(self):
        device = torch.device(self.config['training']['device_name'])
        self.net.to(device)

        summ_writer = SummaryWriter(self.config['training']['summary_dir'])
        multi_pred_weight  = self.config['training'].get('multi_pred_weight', None)
        chpt_prefx  = self.config['training']['checkpoint_prefix']
        iter_start  = self.config['training']['iter_start']
        iter_max    = self.config['training']['iter_max']
        iter_valid  = self.config['training']['iter_valid']
        iter_save   = self.config['training']['iter_save']
        class_num   = self.config['network']['class_num']

        if(iter_start > 0):
            checkpoint_file = "{0:}_{1:}.pt".format(chpt_prefx, iter_start)
            self.checkpoint = torch.load(checkpoint_file, map_location=device)
            assert(self.checkpoint['iteration'] == iter_start)
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            self.checkpoint = None
        self.create_optimizer()

        train_loss      = 0
        train_seg_loss = 0
        train_contour_loss = 0
        train_dice_list = []
        if(self.loss_calculater is None):
            loss_func   = self.config['training']['loss_function']
            self.loss_calculater = SegmentationLossCalculator(loss_func, multi_pred_weight)
        trainIter = iter(self.train_loader)
        print("{0:} training start".format(str(datetime.now())[:-7]))
        threshold_contour = 0.9
        for it in range(iter_start, iter_max):
            try:
                data = next(trainIter)
            except StopIteration:
                trainIter = iter(self.train_loader)
                data = next(trainIter)

            inputs      = self.convert_tensor_type(data['image'])
            labels_prob = self.convert_tensor_type(data['label_prob'])
            contours = self.convert_tensor_type(data["contour"])

            inputs, labels_prob, contours = inputs.to(device), labels_prob.to(device), contours.to(device)
            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            # forward + backward + optimize
            outputs_seg, outputs_contour = self.net(inputs)
            loss_input_dict = {'prediction':outputs_seg, 'ground_truth':labels_prob}
            
            loss_seg   = self.loss_calculater.get_loss(loss_input_dict)
            loss_contour = self.binary_ce_loss(outputs_contour, contours)
            loss = loss_seg + loss_contour
            loss.backward()
            self.optimizer.step()
            self.schedule.step()

            # get dice evaluation for each class
            outputs_seg_argmax = torch.argmax(outputs_seg, dim = 1, keepdim = True)
            dice_list = self.get_dice_argmax(outputs_seg_argmax, labels_prob, class_num)
            train_dice_list.append(dice_list.cpu().numpy())
            outputs_contour_argmax = (torch.sigmoid(outputs_contour) > threshold_contour).float()

            # evaluate performance on validation set
            train_loss = train_loss + loss.item()
            train_seg_loss = train_seg_loss + loss_seg.item()
            train_contour_loss = train_contour_loss + loss_contour.item()
            if (it % iter_valid == iter_valid - 1):
                image = inputs[0, 0: 1, :, :, :].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
                image = make_grid(image, image.size(0), normalize=True)
                summ_writer.add_image("11/image", image, it + 1)
                gt_seg = torch.max(labels_prob[0, :, :, :, :], dim=0, keepdim=True)[1].permute( \
                    1, 0, 2, 3).repeat(1, 3, 1, 1).float()
                gt_seg = make_grid(gt_seg, gt_seg.size(0), normalize=True)
                summ_writer.add_image("11/gt_seg", gt_seg, it + 1)
                gt_contour = contours[0, :, :, :, :].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
                gt_contour = make_grid(gt_contour, gt_contour.size(0))
                summ_writer.add_image("11/gt_contour", gt_contour, it + 1)
                pred_seg = outputs_seg_argmax[0, :, :, :, :].permute(1, 0, 2, 3).repeat(1, 3, 1, 1).float()
                pred_seg = make_grid(pred_seg, pred_seg.size(0), normalize=True)
                summ_writer.add_image("11/pred_seg", pred_seg, it + 1)
                pred_contour = outputs_contour_argmax[0, :, :, :, :].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
                pred_contour = make_grid(pred_contour, pred_contour.size(0))
                summ_writer.add_image("11/pred_contour", pred_contour, it + 1)

                train_avg_loss = train_loss / iter_valid
                train_seg_avg_loss = train_seg_loss / iter_valid
                train_contour_avg_loss = train_contour_loss / iter_valid
                train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
                train_avg_dice = train_cls_dice.mean()
                train_loss = 0.0
                train_seg_loss = 0.0
                train_contour_loss = 0.0
                train_dice_list = []

                valid_loss = 0.0
                valid_seg_loss = 0.0
                valid_contour_loss = 0.0
                valid_dice_list = []
                with torch.no_grad():
                    for data in self.valid_loader:
                        inputs      = self.convert_tensor_type(data['image'])
                        labels_prob = self.convert_tensor_type(data['label_prob'])
                        contours = self.convert_tensor_type(data["contour"])
                        
                        inputs, labels_prob, contours = inputs.to(device), labels_prob.to(device), contours.to(device)
                        outputs_seg, outputs_contour = self.net(inputs)
                        loss_input_dict = {'prediction':outputs_seg, 'ground_truth':labels_prob}

                        loss_seg   = self.loss_calculater.get_loss(loss_input_dict)
                        loss_contour = self.binary_ce_loss(outputs_contour, contours)
                        loss = loss_seg + loss_contour
                        valid_loss = valid_loss + loss.item()
                        valid_seg_loss = valid_seg_loss + loss_seg
                        valid_contour_loss = valid_contour_loss + loss_contour

                        outputs_seg_argmax = torch.argmax(outputs_seg, dim = 1, keepdim = True)
                        dice_list = self.get_dice_argmax(outputs_seg_argmax, labels_prob, class_num)
                        valid_dice_list.append(dice_list.cpu().numpy())
                        outputs_contour_argmax = (torch.sigmoid(outputs_contour) > threshold_contour).float()

                valid_avg_loss = valid_loss / len(self.valid_loader)
                valid_seg_avg_loss = valid_seg_loss / len(self.valid_loader)
                valid_contour_avg_loss = valid_contour_loss / len(self.valid_loader)
                valid_cls_dice = np.asarray(valid_dice_list).mean(axis = 0)
                valid_avg_dice = valid_cls_dice.mean()

                loss_scalers = {'train': train_avg_loss, 'valid': valid_avg_loss}
                summ_writer.add_scalars('loss/loss', loss_scalers, it + 1)
                loss_scalers = {"train": train_seg_avg_loss, "valid": valid_seg_avg_loss}
                summ_writer.add_scalars("loss/loss_seg", loss_scalers, it + 1)
                loss_scalers = {"train": train_contour_avg_loss, "valid": valid_contour_avg_loss}
                summ_writer.add_scalars("loss/loss_contour", loss_scalers, it + 1)

                dice_scalers = {'train': train_avg_dice, 'valid': valid_avg_dice}
                summ_writer.add_scalars('class_avg_dice/origin', dice_scalers, it + 1)
                print('train cls dice', train_cls_dice.shape, train_cls_dice)
                print('valid cls dice', valid_cls_dice.shape, valid_cls_dice)
                for c in range(class_num):
                    dice_scalars = {'train':train_cls_dice[c], 'valid':valid_cls_dice[c]}
                    summ_writer.add_scalars('class_{0:}_dice/origin'.format(c), dice_scalars, it + 1)
                
                print("{0:} it {1:}, loss {2:.4f}, {3:.4f}".format(
                    str(datetime.now())[:-7], it + 1, train_avg_loss, valid_avg_loss))
            if (it + 1 in iter_save):
                save_dict = {'iteration': it + 1,
                             'model_state_dict': self.net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}_{1:}.pt".format(chpt_prefx, it + 1)
                torch.save(save_dict, save_name)    
        summ_writer.close()
    
    def infer(self):
        device = torch.device(self.config['testing']['device_name'])
        self.net.to(device)
        
        if(self.config['testing']['evaluation_mode'] == True):
            self.net.eval()
            if(self.config['testing']['test_time_dropout'] == True):
                def test_time_dropout(m):
                    if(type(m) == nn.Dropout):
                        print('dropout layer')
                        m.train()
                self.net.apply(test_time_dropout)
        output_dir   = self.config['testing']['output_dir']
        class_num    = self.config['network']['class_num']
        mini_batch_size      = self.config['testing']['mini_batch_size']
        mini_patch_inshape   = self.config['testing']['mini_patch_input_shape']
        mini_patch_outshape  = self.config['testing']['mini_patch_output_shape']
        mini_patch_stride    = self.config['testing']['mini_patch_stride']
        output_num       = self.config['testing'].get('output_num', 1)
        label_source = self.config['testing'].get('label_source', None)
        label_target = self.config['testing'].get('label_target', None)
        filename_replace_source = self.config['testing'].get('filename_replace_source', None)
        filename_replace_target = self.config['testing'].get('filename_replace_target', None)
        
        checkpoint_list = self.config["testing"]["checkpoint_list"]
        checkpoint_prefix = self.config["training"]["checkpoint_prefix"]

        infer_time_list = []
        with torch.no_grad():
            start_time = time.time()
            for data in self.test_loder:
                images = self.convert_tensor_type(data['image'])
                names  = data['names']
                print(names[0])
                # ensembe of multiple checkpoints
                predict_list = []
                for checkpoint in checkpoint_list:
                    # load network parameters and set the network as evaluation mode
                    checkpoint = torch.load(checkpoint_prefix + "_%s.pt" % str(checkpoint), map_location=device)
                    self.net.load_state_dict(checkpoint["model_state_dict"])
                    
                    data['predict']  = volume_infer(images, self.net, device, class_num, # (seg, contour)
                        mini_batch_size, mini_patch_inshape, mini_patch_outshape, mini_patch_stride, output_num)
                    data["predict"] = data["predict"][0]  # seg

                    for transform in reversed(range(len(self.transform_list))):
                        if (self.transform_list[transform].inverse):
                            data = self.transform_list[transform].inverse_transform_for_prediction(data)
                    predict_list.append(data["predict"][0])


                predict = np.mean(predict_list, axis=0)
                root_dir  = self.config['dataset']['root_dir']
                save_name = names[0].split('/')[-1]
                if((filename_replace_source is  not None) and (filename_replace_target is not None)):
                    save_name = save_name.replace(filename_replace_source, filename_replace_target)
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                save_name = "{0:}/{1:}".format(output_dir, save_name)

                if self.stage != "ensemble":
                    output = np.asfarray(np.argmax(scipy.special.softmax(predict, axis=0), axis=0), np.uint8)
                    infer_time = time.time() - start_time
                    infer_time_list.append(infer_time)
                    if((label_source is not None) and (label_target is not None)):
                        output = convert_label(output, label_source, label_target)
                    save_nd_array_as_image(output, save_name, root_dir + '/' + names[0])

                if self.stage == "ensemble":
                    if not self.ensemble_dict.__contains__(names[0]):
                        self.ensemble_dict[names[0]] = {}
                        self.ensemble_dict[names[0]]["predict_list"] = []
                        self.ensemble_dict[names[0]]["save_name"] = save_name
                    self.ensemble_dict[names[0]]["predict_list"].append(predict)

        if self.stage != "ensemble":
            infer_time_list = np.asarray(infer_time_list)
            time_avg = infer_time_list.mean()
            time_std = infer_time_list.std()
            print("testing time {0:} +/- {1:}".format(time_avg, time_std))

def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('    python train_infer.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)
    agent    = TrainInferAgentContour(config, stage)
    agent.run()

if __name__ == "__main__":
    main()
    

