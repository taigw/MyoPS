# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import time
import scipy
import random
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
from pymic.io_my.image_read_write import save_nd_array_as_image
from pymic.io_my.nifty_dataset import NiftyDataset
from pymic.io_my.transform3d import get_transform
from pymic.net_run.net_factory import get_network
from pymic.net_run.infer_func import volume_infer
from pymic.net_run.loss import *
from pymic.net_run.get_optimizer import get_optimiser
from pymic.util.image_process import convert_label
from pymic.util.parse_config import parse_config


class TrainInferAgent(object):
    def __init__(self, config, stage = 'train'):
        assert(stage in ['train', 'test', "ensemble"])
        self.config = config
        self.stage  = stage
        self.net    = None
        self.train_set = None 
        self.valid_set = None 
        self.test_set  = None
        self.loss_calculater = None 
        self.tensor_type = config['dataset']['tensor_type']
        
    def set_datasets(self, train_set, valid_set, test_set):
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set  = test_set

    def set_network(self, net):
        self.net = net 

    def set_loss_calculater(self, loss_calculater):
        self.loss_calculater = loss_calculater

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
        dataset  = NiftyDataset(root_dir=root_dir,
                                csv_file  = csv_file,
                                modal_num = modal_num,
                                with_label= not (stage == 'test'),
                                transform = transforms.Compose(self.transform_list))
        return dataset

    def create_dataset(self):
        if(self.stage == 'train'):
            if(self.train_set is None):
                self.train_set = self.get_stage_dataset_from_config('train')
            if(self.valid_set is None):
                self.valid_set = self.get_stage_dataset_from_config('valid')

            batch_size = self.config['training']['batch_size']
            self.train_loader = torch.utils.data.DataLoader(self.train_set, 
                batch_size = batch_size, shuffle=True, num_workers=batch_size * 4)
            self.valid_loader = torch.utils.data.DataLoader(self.valid_set, 
                batch_size = batch_size, shuffle=False, num_workers=batch_size * 4)
        else:
            if(self.test_set  is None):
                self.test_set  = self.get_stage_dataset_from_config('test')
            batch_size = 1
            self.test_loder = torch.utils.data.DataLoader(self.test_set, 
                batch_size=batch_size, shuffle=False, num_workers=batch_size)

    def create_network(self):
        if(self.net is None):
            self.net = get_network(self.config['network'])
        if(self.tensor_type == 'float'):
            self.net.float()
        else:
            self.net.double()
        param_number = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('parameter number:', param_number)

        # from model import BetaVAE_H
        # self.VAE = BetaVAE_H(z_dim=128, nc=4)
        # param_number = sum(p.numel() for p in self.VAE.parameters() if p.requires_grad)
        # self.VAE.load_state_dict(torch.load("/home/c1501/swzhai/projects/MyOPS2020/checkpoint/main/100000")["model_states"]["net"])
        # print("VAE parameter number:", param_number)
        # for para in self.VAE.parameters():
        #     para.requires_grad = False
        
    def create_optimizer(self):
        self.optimizer = get_optimiser(self.config['training']['optimizer'],
                self.net.parameters(),
                self.config['training'])
        # params = list(self.net.parameters()) + list(self.VAE.parameters())
        # self.optimizer = get_optimiser(self.config['training']['optimizer'],
        #         filter(lambda p: p.requires_grad, params),
        #         self.config['training'])
        last_iter = -1
        if(self.checkpoint is not None):
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            last_iter = self.checkpoint['iteration'] - 1
        self.schedule = optim.lr_scheduler.MultiStepLR(self.optimizer,
                self.config['training']['lr_milestones'],
                self.config['training']['lr_gamma'],
                last_epoch = last_iter)

    def convert_tensor_type(self, input_tensor):
        if(self.tensor_type == 'float'):
            return input_tensor.float()
        else:
            return input_tensor.double()

    def print_maxmin(self, *args):
        print("#" * 11 * 11)
        for arg in args:
            print(arg.shape, torch.max(arg), torch.min(arg))
    
    def binary_ce_loss(self, pred, gd, sigmoid=True, weighted=True):
        import torch
        from pymic.net_run.loss import reshape_tensor_to_2D
        if sigmoid:
            pred = torch.sigmoid(pred)
        if weighted:
            num_all = torch.numel(gd)
            num_positive = int(torch.sum(gd))
            num_negative = num_all - num_positive
            coef_pos = num_negative / num_all
            coef_neg = num_positive / num_all
        else:
            coef_pos = coef_neg = 0.5
        pred, gd = reshape_tensor_to_2D(pred), reshape_tensor_to_2D(gd)
        bce = - gd * torch.log(pred + 1e-6) * coef_pos - (1 - gd) * torch.log(1 - pred + 1e-6) * coef_neg
        bce = torch.mean(bce)
        return bce
    
    def get_dice_argmax(self, argmax_map, gd, class_num, convert_onehot=False):
        if convert_onehot:
            gd = get_soft_label(gd, class_num, self.tensor_type)
        soft_out       = get_soft_label(argmax_map, class_num, self.tensor_type)
        soft_out, gd = reshape_prediction_and_ground_truth(soft_out, gd) 
        dice_list = get_classwise_dice(soft_out, gd)
        return dice_list

    def train(self):
        device = torch.device(self.config['training']['device_name'])
        self.net.to(device)
        # self.VAE.to(device)

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
        train_dice_list = []
        if(self.loss_calculater is None):
            loss_func   = self.config['training']['loss_function']
            self.loss_calculater = SegmentationLossCalculator(loss_func, multi_pred_weight)
            if(loss_func == 'noise_robust_dice_loss'):
                self.loss_calculater.set_noise_robust_dice_loss_p(self.config['training']['noise_robust_dice_loss_p'])
        trainIter = iter(self.train_loader)
        print("{0:} training start".format(str(datetime.now())[:-7]))
        for it in range(iter_start, iter_max):
            try:
                data = next(trainIter)
            except StopIteration:
                trainIter = iter(self.train_loader)
                data = next(trainIter)

            inputs      = self.convert_tensor_type(data['image'])
            labels_prob = self.convert_tensor_type(data['label_prob'])

            inputs, labels_prob = inputs.to(device), labels_prob.to(device)
            # zero the parameter gradients
            self.optimizer.zero_grad()
                
            # forward + backward + optimize
            outputs = self.net(inputs)
            loss_input_dict = {'prediction':outputs, 'ground_truth':labels_prob}
            
            loss   = self.loss_calculater.get_loss(loss_input_dict)
            # if it + 1 > 12000:
            #     outputs = F.softmax(outputs, dim=1)
            #     vector_out = self.VAE(outputs)
            #     vector_lab = self.VAE(labels_prob)
            #     loss_mse = F.mse_loss(vector_out, vector_lab)
            #     loss = loss + 0.5 * loss_mse
            loss.backward()
            self.optimizer.step()
            self.schedule.step()

            # get dice evaluation for each class
            outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
            dice_list = self.get_dice_argmax(outputs_argmax, labels_prob, class_num)
            train_dice_list.append(dice_list.cpu().numpy())

            # evaluate performance on validation set
            train_loss = train_loss + loss.item()
            if (it % iter_valid == iter_valid - 1):
                train_avg_loss = train_loss / iter_valid
                train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
                train_avg_dice = train_cls_dice.mean()
                train_loss = 0.0
                train_dice_list = []

                valid_loss = 0.0
                valid_dice_list = []
                with torch.no_grad():
                    for data in self.valid_loader:
                        inputs      = self.convert_tensor_type(data['image'])
                        labels_prob = self.convert_tensor_type(data['label_prob'])
                        
                        inputs, labels_prob = inputs.to(device), labels_prob.to(device)
                        outputs = self.net(inputs)
                        loss_input_dict = {'prediction':outputs, 'ground_truth':labels_prob}
                        if ('label_distance' in data):
                            label_distance = self.convert_tensor_type(data['label_distance'])
                            loss_input_dict['label_distance'] = label_distance.to(device)
                        loss   = self.loss_calculater.get_loss(loss_input_dict)
                        valid_loss = valid_loss + loss.item()

                        outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
                        dice_list = self.get_dice_argmax(outputs_argmax, labels_prob, class_num)
                        valid_dice_list.append(dice_list.cpu().numpy())

                valid_avg_loss = valid_loss / len(self.valid_loader)
                valid_cls_dice = np.asarray(valid_dice_list).mean(axis = 0)
                valid_avg_dice = valid_cls_dice.mean()
                loss_scalers = {'train': train_avg_loss, 'valid': valid_avg_loss}
                summ_writer.add_scalars('loss', loss_scalers, it + 1)
                dice_scalers = {'train': train_avg_dice, 'valid': valid_avg_dice}
                summ_writer.add_scalars('class_avg_dice', dice_scalers, it + 1)
                print('train cls dice', train_cls_dice.shape, train_cls_dice)
                print('valid cls dice', valid_cls_dice.shape, valid_cls_dice)
                for c in range(class_num):
                    dice_scalars = {'train':train_cls_dice[c], 'valid':valid_cls_dice[c]}
                    summ_writer.add_scalars('class_{0:}_dice'.format(c), dice_scalars, it + 1)
                
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
                    
                    data['predict']  = volume_infer(images, self.net, device, class_num, 
                        mini_batch_size, mini_patch_inshape, mini_patch_outshape, mini_patch_stride, output_num)
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

    def ensemble(self):
        self.ensemble_dict = {}
        fold_num = 5
        root_dir = self.config["dataset"]["root_dir"]
        temp_chpt_prefix = self.config["training"]["checkpoint_prefix"]
        # make sure "fold_1" in checkpoint_prefix
        assert "fold_1" in temp_chpt_prefix
        # ensemble of all folds
        for fold in range(fold_num):
            print("#" * 11 * 11)
            print("predicting fold %s" % str(fold + 1))
            self.config["training"]["checkpoint_prefix"] = temp_chpt_prefix.replace("fold_1", "fold_%s" % str(fold + 1))
            self.infer()
        for name in self.ensemble_dict.keys():
            save_name = self.ensemble_dict[name]["save_name"]
            predict_list = self.ensemble_dict[name]["predict_list"]
            predict = np.mean(predict_list, axis=0)
            output = np.asfarray(np.argmax(scipy.special.softmax(predict, axis=0), axis=0), np.uint8)
            save_nd_array_as_image(output, save_name, root_dir + '/' + name)

    def run(self):
        deterministic = self.config["training"]["deterministic"]
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            seed = 1111
            random.seed(seed)
            np.random.seed(seed)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        self.create_dataset()
        self.create_network()
        if(self.stage == 'train'):
            self.train()
        elif(self.stage == "test"):
            self.infer()
        elif(self.stage == "ensemble"):
            self.ensemble()
        else:
            print("stage is not correct")

def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('    python train_infer.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)
    agent    = TrainInferAgent(config, stage)
    agent.run()

if __name__ == "__main__":
    main()
    

