"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, IdDis
from reIDmodel import ft_net, ft_netABe
from utils import get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import torchvision
import copy
import os
import cv2
import numpy as np
from random_erasing import RandomErasing
from shutil import copyfile, copytree
import random
import yaml
from  re_ranking_one import re_ranking_one
from sklearn.cluster import DBSCAN



def to_gray(half=False): #simple
    def forward(x):
        x = torch.mean(x, dim=1, keepdim=True)
        if half:
            x = x.half()
        return x
    return forward

def to_edge(x):
    x = x.data.cpu()
    out = torch.FloatTensor(x.size(0), x.size(2), x.size(3))
    for i in range(x.size(0)):
        xx = recover(x[i,:,:,:])   # 3 channel, 256x128x3
        xx = cv2.cvtColor(xx, cv2.COLOR_RGB2GRAY) # 256x128x1
        xx = cv2.Canny(xx, 10, 200) #256x128
        xx = xx/255.0 - 0.5 # {-0.5,0.5}
        xx += np.random.randn(xx.shape[0],xx.shape[1])*0.1  #add random noise
        xx = torch.from_numpy(xx.astype(np.float32))
        out[i,:,:] = xx
    out = out.unsqueeze(1) 
    return out.cuda()

def scale2(x):
    if x.size(2) > 128: # do not need to scale the input
        return x
    x = torch.nn.functional.upsample(x, scale_factor=2, mode='nearest')  #bicubic is not available for the time being.
    return x

def recover(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    inp = inp.astype(np.uint8)
    return inp

def train_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def update_teacher(model_s, model_t, alpha=0.999):
    for param_s, param_t in zip(model_s.parameters(), model_t.parameters()):
        param_t.data.mul_(alpha).add_(1 - alpha, param_s.data)

def predict_label(teacher_models, inputs, num_class, alabel, slabel, teacher_style=0):
# teacher_style:
# 0: Our smooth dynamic label
# 1: Pseudo label, hard dynamic label
# 2: Conditional label, hard static label 
# 3: LSRO, static smooth label
# 4: Dynamic Soft Two-label
# alabel is appearance label
    if teacher_style == 0:
        count = 0
        sm = nn.Softmax(dim=1)
        for teacher_model in teacher_models:
            _, outputs_t1 = teacher_model(inputs) 
            outputs_t1 = sm(outputs_t1.detach())
            _, outputs_t2 = teacher_model(fliplr(inputs)) 
            outputs_t2 = sm(outputs_t2.detach())
            if count==0:
                outputs_t = outputs_t1 + outputs_t2
            else:
                outputs_t = outputs_t * opt.alpha  # old model decay
                outputs_t += outputs_t1 + outputs_t2
            count +=2
    elif teacher_style == 1:  # dynamic one-hot  label
        count = 0
        sm = nn.Softmax(dim=1)
        for teacher_model in teacher_models:
            _, outputs_t1 = teacher_model(inputs)
            outputs_t1 = sm(outputs_t1.detach())  # change softmax to max
            _, outputs_t2 = teacher_model(fliplr(inputs))
            outputs_t2 = sm(outputs_t2.detach())
            if count==0:
                outputs_t = outputs_t1 + outputs_t2
            else:
                outputs_t = outputs_t * opt.alpha  # old model decay
                outputs_t += outputs_t1 + outputs_t2
            count +=2
        _, dlabel = torch.max(outputs_t.data, 1)
        outputs_t = torch.zeros(inputs.size(0), num_class).cuda()
        for i in range(inputs.size(0)):
            outputs_t[i, dlabel[i]] = 1
    elif teacher_style == 2: # appearance label
        outputs_t = torch.zeros(inputs.size(0), num_class).cuda()
        for i in range(inputs.size(0)):
            outputs_t[i, alabel[i]] = 1
    elif teacher_style == 3: # LSRO
        outputs_t = torch.ones(inputs.size(0), num_class).cuda()
    elif teacher_style == 4: #Two-label
        count = 0
        sm = nn.Softmax(dim=1)
        for teacher_model in teacher_models:
            _, outputs_t1 = teacher_model(inputs)
            outputs_t1 = sm(outputs_t1.detach())
            _, outputs_t2 = teacher_model(fliplr(inputs))
            outputs_t2 = sm(outputs_t2.detach())
            if count==0:
                outputs_t = outputs_t1 + outputs_t2
            else:
                outputs_t = outputs_t * opt.alpha  # old model decay
                outputs_t += outputs_t1 + outputs_t2
            count +=2
        mask = torch.zeros(outputs_t.shape)
        mask = mask.cuda()
        for i in range(inputs.size(0)):
            mask[i, alabel[i]] = 1
            mask[i, slabel[i]] = 1
        outputs_t = outputs_t*mask
    else:
        print('not valid style. teacher-style is in [0-3].')

    s = torch.sum(outputs_t, dim=1, keepdim=True)
    s = s.expand_as(outputs_t)
    outputs_t = outputs_t/s
    return outputs_t

######################################################################
# Load model
#---------------------------
def load_network(network, name):
   save_path = os.path.join('./models',name,'net_last.pth')
   network.load_state_dict(torch.load(save_path))
   return network

def load_config(name):
    config_path = os.path.join('./models',name,'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)
    return config

def norm(f, dim = 1):
    f = f.squeeze()
    fnorm = torch.norm(f, p=2, dim=dim, keepdim=True)
    f = f.div(fnorm.expand_as(f))
    return f

def get_id(img_path, time_constraint = False):
    camera_id = []
    time_id = []
    labels = []
    for path, v in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if time_constraint:
            metadata = filename.split('_')
            num_metadata = len(metadata)
            if num_metadata == 3:
                time = filename.split('f')[1]
            elif num_metadata == 4:
                time = metadata[2]
        # print(camera)
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
        if time_constraint:
            if num_metadata == 3:
                time_id.append(int(time[0:7]))
            elif num_metadata == 4:
                time_id.append(int(time[0:6]))
    return camera_id, labels, time_id

def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    # same camera
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, gc, good_index, junk_index)
    return CMC_tmp

def compute_mAP(index, gc, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    ranked_camera = gc[index]
    mask = np.in1d(index, junk_index, invert=True)
    mask2 = np.in1d(index, np.append(good_index, junk_index), invert=True)
    index = index[mask]
    ranked_camera = ranked_camera[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc

class DGNetpp_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(DGNetpp_Trainer, self).__init__()
        lr_g = hyperparameters['lr_g']
        lr_d = hyperparameters['lr_d']
        lr_id_d = hyperparameters['lr_id_d']
        ID_class_a = hyperparameters['ID_class_a']

        # Initiate the networks
        # We do not need to manually set fp16 in the network. So here I set fp16=False.
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'], fp16=False)  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'], fp16=False)  # auto-encoder for domain b

        if not 'ID_stride' in hyperparameters.keys():
            hyperparameters['ID_stride'] = 2

        self.id_a = ft_netABe(ID_class_a + hyperparameters['ID_class_b'], stride=hyperparameters['ID_stride'], norm=hyperparameters['norm_id'], pool=hyperparameters['pool'])

        self.id_b = self.id_a
        self.dis_a = MsImageDis(3, hyperparameters['dis'], fp16=False)  # discriminator for domain a
        self.dis_b = self.dis_a

        self.id_dis = IdDis(hyperparameters['gen']['id_dim'], hyperparameters['dis'], fp16=False)  # ID discriminator

        # load teachers
        if hyperparameters['teacher'] != "":
            teacher_name = hyperparameters['teacher']
            print(teacher_name)
            teacher_names = teacher_name.split(',')
            teacher_model = nn.ModuleList()
            teacher_count = 0
            for teacher_name in teacher_names:
                config_tmp = load_config(teacher_name)
                if 'stride' in config_tmp:
                    stride = config_tmp['stride']
                else:
                    stride = 2
                model_tmp = ft_net(ID_class_a, stride = stride)
                teacher_model_tmp = load_network(model_tmp, teacher_name)
                teacher_model_tmp.model.fc = nn.Sequential()  # remove the original fc layer in ImageNet
                teacher_model_tmp = teacher_model_tmp.cuda()
                teacher_model.append(teacher_model_tmp.cuda().eval())
                teacher_count += 1
            self.teacher_model = teacher_model
            if hyperparameters['train_bn']:
                self.teacher_model = self.teacher_model.apply(train_bn)

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        display_size = int(hyperparameters['display_size'])

        # RGB to one channel
        if hyperparameters['single'] == 'edge':
            self.single = to_edge
        else:
            self.single = to_gray(False)

        # Random Erasing when training
        if not 'erasing_p' in hyperparameters.keys():
            hyperparameters['erasing_p'] = 0
        self.single_re = RandomErasing(probability=hyperparameters['erasing_p'], mean=[0.0, 0.0, 0.0])

        if not 'T_w' in hyperparameters.keys():
            hyperparameters['T_w'] = 1
        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_a_params = list(self.dis_a.parameters())
        gen_a_params = list(self.gen_a.parameters())
        gen_b_params = list(self.gen_b.parameters())
        id_dis_params = list(self.id_dis.parameters())

        self.dis_a_opt = torch.optim.Adam([p for p in dis_a_params if p.requires_grad],
                                        lr=lr_d, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.id_dis_opt = torch.optim.Adam([p for p in id_dis_params if p.requires_grad],
                                        lr=lr_id_d, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_a_opt = torch.optim.Adam([p for p in gen_a_params if p.requires_grad],
                                        lr=lr_g, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_b_opt = torch.optim.Adam([p for p in gen_b_params if p.requires_grad],
                                        lr=lr_g, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        # id params
        ignored_params = (list(map(id, self.id_a.classifier1.parameters()))
                          + list(map(id, self.id_a.classifier2.parameters())))
        base_params = filter(lambda p: id(p) not in ignored_params, self.id_a.parameters())
        lr2 = hyperparameters['lr2']
        self.id_opt = torch.optim.SGD([
            {'params': base_params, 'lr': lr2},
            {'params': self.id_a.classifier1.parameters(), 'lr': lr2 * 10},
            {'params': self.id_a.classifier2.parameters(), 'lr': lr2 * 10}
        ], weight_decay=hyperparameters['weight_decay'], momentum=0.9, nesterov=True)

        self.dis_a_scheduler = get_scheduler(self.dis_a_opt, hyperparameters)
        self.id_dis_scheduler = get_scheduler(self.id_dis_opt, hyperparameters)
        self.id_dis_scheduler.gamma = hyperparameters['gamma2']
        self.gen_a_scheduler = get_scheduler(self.gen_a_opt, hyperparameters)
        self.gen_b_scheduler = get_scheduler(self.gen_b_opt, hyperparameters)
        self.id_scheduler = get_scheduler(self.id_opt, hyperparameters)
        self.id_scheduler.gamma = hyperparameters['gamma2']

        # ID Loss
        self.id_criterion = nn.CrossEntropyLoss()
        self.criterion_teacher = nn.KLDivLoss(size_average=False)
        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def to_re(self, x):
        out = torch.FloatTensor(x.size(0), x.size(1), x.size(2), x.size(3))
        out = out.cuda()
        for i in range(x.size(0)):
            out[i, :, :, :] = self.single_re(x[i, :, :, :])
        return out

    def recon_criterion(self, input, target):
        diff = input - target.detach()
        return torch.mean(torch.abs(diff[:]))

    def recon_criterion_sqrt(self, input, target):
        diff = input - target
        return torch.mean(torch.sqrt(torch.abs(diff[:]) + 1e-8))

    def recon_criterion2(self, input, target):
        diff = input - target
        return torch.mean(diff[:] ** 2)

    def recon_cos(self, input, target):
        cos = torch.nn.CosineSimilarity()
        cos_dis = 1 - cos(input, target)
        return torch.mean(cos_dis[:])

    def forward(self, x_a, x_b):
        self.eval()
        s_a = self.gen_a.encode(self.single(x_a))
        s_b = self.gen_b.encode(self.single(x_b))
        f_a, _, _ = self.id_a(scale2(x_a))
        f_b, _, _ = self.id_b(scale2(x_b))
        x_ba = self.gen_b.decode(s_b, f_a)
        x_ab = self.gen_a.decode(s_a, f_b)
        self.train()
        return x_ab, x_ba

    def gen_update_ab(self, x_a, l_a, xp_a, x_b, l_b, xp_b, hyperparameters, iteration):
        # ppa, ppb is the same person
        self.gen_a_opt.zero_grad()
        self.gen_b_opt.zero_grad()
        self.id_opt.zero_grad()
        self.id_dis_opt.zero_grad()
        # encode
        s_a = self.gen_a.encode(self.single(x_a))
        s_b = self.gen_b.encode(self.single(x_b))
        f_a, p_a, fe_a = self.id_a(scale2(x_a))
        f_b, p_b, fe_b = self.id_b(scale2(x_b))
        # autodecode
        x_a_recon = self.gen_a.decode(s_a, f_a)
        x_b_recon = self.gen_b.decode(s_b, f_b)

        # encode the same ID different photo
        fp_a, pp_a, fe_pa = self.id_a(scale2(xp_a))
        fp_b, pp_b, fe_pb = self.id_b(scale2(xp_b))

        # decode the same person
        x_a_recon_p = self.gen_a.decode(s_a, fp_a)
        x_b_recon_p = self.gen_b.decode(s_b, fp_b)

        # has gradient
        x_ba = self.gen_b.decode(s_b, f_a)
        x_ab = self.gen_a.decode(s_a, f_b)
        # no gradient
        x_ba_copy = Variable(x_ba.data, requires_grad=False)
        x_ab_copy = Variable(x_ab.data, requires_grad=False)

        rand_num = random.uniform(0, 1)
        #################################
        # encode structure
        if hyperparameters['use_encoder_again'] >= rand_num:
            # encode again (encoder is tuned, input is fixed)
            s_a_recon = self.gen_a.enc_content(self.single(x_ab_copy))
            s_b_recon = self.gen_b.enc_content(self.single(x_ba_copy))
        else:
            # copy the encoder
            self.enc_content_a_copy = copy.deepcopy(self.gen_a.enc_content)
            self.enc_content_a_copy = self.enc_content_a_copy.eval()
            self.enc_content_b_copy = copy.deepcopy(self.gen_b.enc_content)
            self.enc_content_b_copy = self.enc_content_b_copy.eval()
            # encode again (encoder is fixed, input is tuned)
            s_a_recon = self.enc_content_a_copy(self.single(x_ab))
            s_b_recon = self.enc_content_b_copy(self.single(x_ba))

        #################################
        # encode appearance
        self.id_a_copy = copy.deepcopy(self.id_a)
        self.id_a_copy = self.id_a_copy.eval()
        if hyperparameters['train_bn']:
            self.id_a_copy = self.id_a_copy.apply(train_bn)
        self.id_b_copy = self.id_a_copy
        # encode again (encoder is fixed, input is tuned)
        f_a_recon, p_a_recon, _ = self.id_a_copy(scale2(x_ba))
        f_b_recon, p_b_recon, _ = self.id_b_copy(scale2(x_ab))

        # teacher Loss
        #  Tune the ID model
        log_sm = nn.LogSoftmax(dim=1)
        if hyperparameters['teacher_w'] > 0 and hyperparameters['teacher'] != "":
            if hyperparameters['ID_style'] == 'normal':
                _, p_a_student, _ = self.id_a(scale2(x_ba_copy))
                p_a_student = log_sm(p_a_student)
                p_a_teacher = predict_label(self.teacher_model, scale2(x_ba_copy))
                self.loss_teacher = self.criterion_teacher(p_a_student, p_a_teacher) / p_a_student.size(0)

                _, p_b_student, _ = self.id_b(scale2(x_ab_copy))
                p_b_student = log_sm(p_b_student)
                p_b_teacher = predict_label(self.teacher_model, scale2(x_ab_copy))
                self.loss_teacher += self.criterion_teacher(p_b_student, p_b_teacher) / p_b_student.size(0)
            elif hyperparameters['ID_style'] == 'AB':
                # normal teacher-student loss
                # BA -> LabelA(smooth) + LabelB(batchB)
                _, p_ba_student, _ = self.id_a(scale2(x_ba_copy))  # f_a, s_b
                p_a_student = log_sm(p_ba_student[0])
                with torch.no_grad():
                    p_a_teacher = predict_label(self.teacher_model, scale2(x_ba_copy),
                                                num_class=hyperparameters['ID_class_a'], alabel=l_a, slabel=l_b,
                                                teacher_style=hyperparameters['teacher_style'])
                p_a_teacher = torch.cat(
                    (p_a_teacher, torch.zeros((p_a_teacher.size(0), hyperparameters['ID_class_b'])).cuda()),
                    1).detach()
                self.loss_teacher = self.criterion_teacher(p_a_student, p_a_teacher) / p_a_student.size(0)

                _, p_ab_student, _ = self.id_b(scale2(x_ab_copy))  # f_b, s_a
                # branch b loss
                # here we give different label
                self.loss_teacher = hyperparameters['T_w'] * self.loss_teacher
                loss_B = self.id_criterion(p_ab_student[1], l_a)
                self.loss_teacher = self.loss_teacher + hyperparameters['B_w'] * loss_B
        else:
            self.loss_teacher = 0.0

        # decode again (if needed)
        if hyperparameters['use_decoder_again']:
            x_aba = self.gen_a.decode(s_a_recon, f_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
            x_bab = self.gen_b.decode(s_b_recon, f_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        else:
            self.mlp_w_a_copy = copy.deepcopy(self.gen_a.mlp_w)
            self.mlp_b_a_copy = copy.deepcopy(self.gen_a.mlp_b)
            self.dec_a_copy = copy.deepcopy(self.gen_a.dec)  # Error
            ID = f_a_recon
            ID_Style = ID.view(ID.shape[0], ID.shape[1], 1, 1)
            adain_params_w_a = self.mlp_w_a_copy(ID_Style)
            adain_params_b_a = self.mlp_b_a_copy(ID_Style)
            self.gen_a.assign_adain_params(adain_params_w_a, adain_params_b_a, self.dec_a_copy)
            x_aba = self.dec_a_copy(s_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

            self.mlp_w_b_copy = copy.deepcopy(self.gen_b.mlp_w)
            self.mlp_b_b_copy = copy.deepcopy(self.gen_b.mlp_b)
            self.dec_b_copy = copy.deepcopy(self.gen_b.dec)  # Error
            ID = f_b_recon
            ID_Style = ID.view(ID.shape[0], ID.shape[1], 1, 1)
            adain_params_w_b = self.mlp_w_b_copy(ID_Style)
            adain_params_b_b = self.mlp_b_b_copy(ID_Style)
            self.gen_a.assign_adain_params(adain_params_w_b, adain_params_b_b, self.dec_b_copy)
            x_bab = self.dec_b_copy(s_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # auto-encoder image reconstruction
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_xp_a = self.recon_criterion(x_a_recon_p, x_a)
        self.loss_gen_recon_xp_b = self.recon_criterion(x_b_recon_p, x_b)

        # feature reconstruction
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_f_a = self.recon_criterion(f_a_recon, f_a) if hyperparameters['recon_f_w'] > 0 else 0
        self.loss_gen_recon_f_b = self.recon_criterion(f_b_recon, f_b) if hyperparameters['recon_f_w'] > 0 else 0

        # Random Erasing only effect the ID and PID loss.
        if hyperparameters['erasing_p'] > 0:
            x_a_re = self.to_re(scale2(x_a.clone()))
            x_b_re = self.to_re(scale2(x_b.clone()))
            xp_a_re = self.to_re(scale2(xp_a.clone()))
            xp_b_re = self.to_re(scale2(xp_b.clone()))
            _, p_a, _ = self.id_a(x_a_re)
            _, p_b, _ = self.id_b(x_b_re)
            # encode the same ID different photo
            _, pp_a, _ = self.id_a(xp_a_re)
            _, pp_b, _ = self.id_b(xp_b_re)

        # ID loss AND Tune the Generated image
        weight_B = hyperparameters['teacher_w'] * hyperparameters['B_w']
        if hyperparameters['id_tgt']:
            self.loss_id = self.id_criterion(p_a[0], l_a) + self.id_criterion(p_b[0], l_b) \
                           + weight_B * (self.id_criterion(p_a[1], l_a) + self.id_criterion(p_b[1], l_b))
            self.loss_pid = self.id_criterion(pp_a[0], l_a) + hyperparameters['tgt_pos'] * self.id_criterion(pp_b[0],l_b)  # + weight_B * ( self.id_criterion(pp_a[1], l_a) + self.id_criterion(pp_b[1], l_b) )
            self.loss_gen_recon_id = self.id_criterion(p_a_recon[0], l_a) + self.id_criterion(p_b_recon[0], l_b)
        else:
            self.loss_id = self.id_criterion(p_a[0], l_a) + weight_B * self.id_criterion(p_a[1], l_a)
            self.loss_pid = self.id_criterion(pp_a[0], l_a)
            self.loss_gen_recon_id = self.id_criterion(p_a_recon[0], l_a)

        # print(f_a_recon, f_a)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_a.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0

        # ID domain adversarial loss
        self.loss_gen_id_adv = self.id_dis.calc_gen_loss(fe_b) if hyperparameters['id_adv_w'] > 0 else 0

        if iteration > hyperparameters['warm_iter']:
            hyperparameters['recon_f_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_f_w'] = min(hyperparameters['recon_f_w'], hyperparameters['max_w'])
            hyperparameters['recon_s_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_s_w'] = min(hyperparameters['recon_s_w'], hyperparameters['max_w'])
            hyperparameters['recon_x_cyc_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_x_cyc_w'] = min(hyperparameters['recon_x_cyc_w'], hyperparameters['max_cyc_w'])

        if iteration > hyperparameters['warm_teacher_iter']:
            hyperparameters['teacher_w'] += hyperparameters['warm_scale']
            hyperparameters['teacher_w'] = min(hyperparameters['teacher_w'], hyperparameters['max_teacher_w'])

        hyperparameters['id_adv_w'] += hyperparameters['adv_warm_scale']
        hyperparameters['id_adv_w'] = min(hyperparameters['id_adv_w'], hyperparameters['id_adv_w_max'])
        
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_xp_w'] * self.loss_gen_recon_xp_a + \
                              hyperparameters['recon_f_w'] * self.loss_gen_recon_f_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_xp_w'] * hyperparameters['recon_xp_tgt_w'] * self.loss_gen_recon_xp_b + \
                              hyperparameters['recon_f_w'] * self.loss_gen_recon_f_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['id_w'] * self.loss_id + \
                              hyperparameters['pid_w'] * self.loss_pid + \
                              hyperparameters['recon_id_w'] * self.loss_gen_recon_id + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
                              hyperparameters['teacher_w'] * self.loss_teacher + \
                              hyperparameters['id_adv_w'] * self.loss_gen_id_adv
        self.loss_gen_total.backward()
        self.gen_a_opt.step()
        self.gen_b_opt.step()
        self.id_opt.step()
        print(
            "L_total: %.4f, L_gan: %.4f, L_adv: %.4f, Lx: %.4f, Lxp: %.4f, Lrecycle:%.4f, Lf: %.4f, Ls: %.4f, Recon-id: %.4f, id: %.4f, pid:%.4f, teacher: %.4f" % (
            self.loss_gen_total, \
            hyperparameters['gan_w'] * (self.loss_gen_adv_a + self.loss_gen_adv_b), \
            hyperparameters['id_adv_w'] * (self.loss_gen_id_adv), \
            hyperparameters['recon_x_w'] * (self.loss_gen_recon_x_a + self.loss_gen_recon_x_b), \
            hyperparameters['recon_xp_w'] * (self.loss_gen_recon_xp_a + hyperparameters['recon_xp_tgt_w'] * self.loss_gen_recon_xp_b), \
            hyperparameters['recon_x_cyc_w'] * (self.loss_gen_cycrecon_x_a + self.loss_gen_cycrecon_x_b), \
            hyperparameters['recon_f_w'] * (self.loss_gen_recon_f_a + self.loss_gen_recon_f_b), \
            hyperparameters['recon_s_w'] * (self.loss_gen_recon_s_a + self.loss_gen_recon_s_b), \
            hyperparameters['recon_id_w'] * self.loss_gen_recon_id, \
            hyperparameters['id_w'] * self.loss_id, \
            hyperparameters['pid_w'] * self.loss_pid, \
            hyperparameters['teacher_w'] * self.loss_teacher))

    def gen_update_aa(self, x_a, l_a, xp_a, x_b, l_b, xp_b, hyperparameters, iteration):
        # ppa, ppb is the same person
        self.gen_a_opt.zero_grad()
        self.id_opt.zero_grad()
        self.id_dis_opt.zero_grad()
        # encode
        s_a = self.gen_a.encode(self.single(x_a))
        s_b = self.gen_a.encode(self.single(x_b))
        f_a, p_a, _ = self.id_a(scale2(x_a))
        f_b, p_b, _ = self.id_a(scale2(x_b))
        # autodecode
        x_a_recon = self.gen_a.decode(s_a, f_a)
        x_b_recon = self.gen_a.decode(s_b, f_b)

        # encode the same ID different photo
        fp_a, pp_a, _ = self.id_a(scale2(xp_a))
        fp_b, pp_b, _ = self.id_a(scale2(xp_b))

        # decode the same person
        x_a_recon_p = self.gen_a.decode(s_a, fp_a)
        x_b_recon_p = self.gen_a.decode(s_b, fp_b)

        # has gradient
        x_ba = self.gen_a.decode(s_b, f_a)
        x_ab = self.gen_a.decode(s_a, f_b)
        # no gradient
        x_ba_copy = Variable(x_ba.data, requires_grad=False)
        x_ab_copy = Variable(x_ab.data, requires_grad=False)

        rand_num = random.uniform(0, 1)
        #################################
        # encode structure
        if hyperparameters['use_encoder_again'] >= rand_num:
            # encode again (encoder is tuned, input is fixed)
            s_a_recon = self.gen_a.enc_content(self.single(x_ab_copy))
            s_b_recon = self.gen_a.enc_content(self.single(x_ba_copy))
        else:
            # copy the encoder
            self.enc_content_copy = copy.deepcopy(self.gen_a.enc_content)
            self.enc_content_copy = self.enc_content_copy.eval()
            # encode again (encoder is fixed, input is tuned)
            s_a_recon = self.enc_content_copy(self.single(x_ab))
            s_b_recon = self.enc_content_copy(self.single(x_ba))

        #################################
        # encode appearance
        self.id_a_copy = copy.deepcopy(self.id_a)
        self.id_a_copy = self.id_a_copy.eval()
        if hyperparameters['train_bn']:
            self.id_a_copy = self.id_a_copy.apply(train_bn)
        self.id_b_copy = self.id_a_copy
        # encode again (encoder is fixed, input is tuned)
        f_a_recon, p_a_recon, _ = self.id_a_copy(scale2(x_ba))
        f_b_recon, p_b_recon, _ = self.id_b_copy(scale2(x_ab))

        # teacher Loss
        #  Tune the ID model
        log_sm = nn.LogSoftmax(dim=1)
        if hyperparameters['teacher_w'] > 0 and hyperparameters['teacher'] != "":
            if hyperparameters['ID_style'] == 'normal':
                _, p_a_student, _ = self.id_a(scale2(x_ba_copy))
                p_a_student = log_sm(p_a_student)
                p_a_teacher = predict_label(self.teacher_model, scale2(x_ba_copy))
                self.loss_teacher = self.criterion_teacher(p_a_student, p_a_teacher) / p_a_student.size(0)

                _, p_b_student, _ = self.id_a(scale2(x_ab_copy))
                p_b_student = log_sm(p_b_student)
                p_b_teacher = predict_label(self.teacher_model, scale2(x_ab_copy))
                self.loss_teacher += self.criterion_teacher(p_b_student, p_b_teacher) / p_b_student.size(0)
            elif hyperparameters['ID_style'] == 'AB':
                # normal teacher-student loss
                # BA -> LabelA(smooth) + LabelB(batchB)
                _, p_ba_student, _ = self.id_a(scale2(x_ba_copy))  # f_a, s_b
                p_a_student = log_sm(p_ba_student[0])
                with torch.no_grad():
                    p_a_teacher = predict_label(self.teacher_model, scale2(x_ba_copy),
                                                num_class=hyperparameters['ID_class_a'], alabel=l_a, slabel=l_b,
                                                teacher_style=hyperparameters['teacher_style'])
                p_a_teacher = torch.cat(
                    (p_a_teacher, torch.zeros((p_a_teacher.size(0), hyperparameters['ID_class_b'])).cuda()),
                    1).detach()
                self.loss_teacher = self.criterion_teacher(p_a_student, p_a_teacher) / p_a_student.size(0)

                _, p_ab_student, _ = self.id_a(scale2(x_ab_copy))  # f_b, s_a
                p_b_student = log_sm(p_ab_student[0])
                with torch.no_grad():
                    p_b_teacher = predict_label(self.teacher_model, scale2(x_ab_copy),
                                                num_class=hyperparameters['ID_class_a'], alabel=l_b, slabel=l_a,
                                                teacher_style=hyperparameters['teacher_style'])
                p_b_teacher = torch.cat(
                    (p_b_teacher, torch.zeros((p_b_teacher.size(0), hyperparameters['ID_class_b'])).cuda()),
                    1).detach()
                self.loss_teacher += self.criterion_teacher(p_b_student, p_b_teacher) / p_b_student.size(0)

                # branch b loss
                # here we give different label
                loss_B = self.id_criterion(p_ba_student[1], l_b) + self.id_criterion(p_ab_student[1], l_a)
                self.loss_teacher = hyperparameters['T_w'] * self.loss_teacher + hyperparameters['B_w'] * loss_B
        else:
            self.loss_teacher = 0.0

        # decode again (if needed)
        if hyperparameters['use_decoder_again']:
            x_aba = self.gen_a.decode(s_a_recon, f_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
            x_bab = self.gen_a.decode(s_b_recon, f_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        else:
            self.mlp_w_copy = copy.deepcopy(self.gen_a.mlp_w)
            self.mlp_b_copy = copy.deepcopy(self.gen_a.mlp_b)
            self.dec_copy = copy.deepcopy(self.gen_a.dec)  # Error
            ID = f_a_recon
            ID_Style = ID.view(ID.shape[0], ID.shape[1], 1, 1)
            adain_params_w = self.mlp_w_copy(ID_Style)
            adain_params_b = self.mlp_b_copy(ID_Style)
            self.gen_a.assign_adain_params(adain_params_w, adain_params_b, self.dec_copy)
            x_aba = self.dec_copy(s_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

            ID = f_b_recon
            ID_Style = ID.view(ID.shape[0], ID.shape[1], 1, 1)
            adain_params_w = self.mlp_w_copy(ID_Style)
            adain_params_b = self.mlp_b_copy(ID_Style)
            self.gen_a.assign_adain_params(adain_params_w, adain_params_b, self.dec_copy)
            x_bab = self.dec_copy(s_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # auto-encoder image reconstruction
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_xp_a = self.recon_criterion(x_a_recon_p, x_a)
        self.loss_gen_recon_xp_b = self.recon_criterion(x_b_recon_p, x_b)

        # feature reconstruction
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_f_a = self.recon_criterion(f_a_recon, f_a) if hyperparameters['recon_f_w'] > 0 else 0
        self.loss_gen_recon_f_b = self.recon_criterion(f_b_recon, f_b) if hyperparameters['recon_f_w'] > 0 else 0

        # Random Erasing only effect the ID and PID loss.
        if hyperparameters['erasing_p'] > 0:
            x_a_re = self.to_re(scale2(x_a.clone()))
            x_b_re = self.to_re(scale2(x_b.clone()))
            xp_a_re = self.to_re(scale2(xp_a.clone()))
            xp_b_re = self.to_re(scale2(xp_b.clone()))
            _, p_a, _ = self.id_a(x_a_re)
            _, p_b, _ = self.id_a(x_b_re)
            # encode the same ID different photo
            _, pp_a, _ = self.id_a(xp_a_re)
            _, pp_b, _ = self.id_a(xp_b_re)

        # ID loss AND Tune the Generated image
        weight_B = hyperparameters['teacher_w'] * hyperparameters['B_w']
        self.loss_id = self.id_criterion(p_a[0], l_a) + self.id_criterion(p_b[0], l_b) \
                       + weight_B * (self.id_criterion(p_a[1], l_a) + self.id_criterion(p_b[1], l_b))
        self.loss_pid = self.id_criterion(pp_a[0], l_a) + self.id_criterion(pp_b[0],
                                                                            l_b)  # + weight_B * ( self.id_criterion(pp_a[1], l_a) + self.id_criterion(pp_b[1], l_b) )
        self.loss_gen_recon_id = self.id_criterion(p_a_recon[0], l_a) + self.id_criterion(p_b_recon[0], l_b)

        # print(f_a_recon, f_a)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_a.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0

        # ID domain adversarial loss
        self.loss_gen_id_adv = 0.0

        if iteration > hyperparameters['warm_iter']:
            hyperparameters['recon_f_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_f_w'] = min(hyperparameters['recon_f_w'], hyperparameters['max_w'])
            hyperparameters['recon_s_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_s_w'] = min(hyperparameters['recon_s_w'], hyperparameters['max_w'])
            hyperparameters['recon_x_cyc_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_x_cyc_w'] = min(hyperparameters['recon_x_cyc_w'], hyperparameters['max_cyc_w'])

        if iteration > hyperparameters['warm_teacher_iter']:
            hyperparameters['teacher_w'] += hyperparameters['warm_scale']
            hyperparameters['teacher_w'] = min(hyperparameters['teacher_w'], hyperparameters['max_teacher_w'])
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_xp_w'] * self.loss_gen_recon_xp_a + \
                              hyperparameters['recon_f_w'] * self.loss_gen_recon_f_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_xp_w'] * self.loss_gen_recon_xp_b + \
                              hyperparameters['recon_f_w'] * self.loss_gen_recon_f_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['id_w'] * self.loss_id + \
                              hyperparameters['pid_w'] * self.loss_pid + \
                              hyperparameters['recon_id_w'] * self.loss_gen_recon_id + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
                              hyperparameters['teacher_w'] * self.loss_teacher

        self.loss_gen_total.backward()
        self.gen_a_opt.step()
        self.id_opt.step()
        print(
            "L_total: %.4f, L_gan: %.4f, L_adv: %.4f, Lx: %.4f, Lxp: %.4f, Lrecycle:%.4f, Lf: %.4f, Ls: %.4f, Recon-id: %.4f, id: %.4f, pid:%.4f, teacher: %.4f" % (
            self.loss_gen_total, \
            hyperparameters['gan_w'] * (self.loss_gen_adv_a + self.loss_gen_adv_b), \
            hyperparameters['id_adv_w'] * (self.loss_gen_id_adv), \
            hyperparameters['recon_x_w'] * (self.loss_gen_recon_x_a + self.loss_gen_recon_x_b), \
            hyperparameters['recon_xp_w'] * (self.loss_gen_recon_xp_a + self.loss_gen_recon_xp_b), \
            hyperparameters['recon_x_cyc_w'] * (self.loss_gen_cycrecon_x_a + self.loss_gen_cycrecon_x_b), \
            hyperparameters['recon_f_w'] * (self.loss_gen_recon_f_a + self.loss_gen_recon_f_b), \
            hyperparameters['recon_s_w'] * (self.loss_gen_recon_s_a + self.loss_gen_recon_s_b), \
            hyperparameters['recon_id_w'] * self.loss_gen_recon_id, \
            hyperparameters['id_w'] * self.loss_id, \
            hyperparameters['pid_w'] * self.loss_pid, \
            hyperparameters['teacher_w'] * self.loss_teacher))

    def gen_update_bb(self, x_a, l_a, xp_a, x_b, l_b, xp_b, hyperparameters, iteration):
        # ppa, ppb is the same person
        self.gen_b_opt.zero_grad()
        self.id_opt.zero_grad()
        self.id_dis_opt.zero_grad()
        # encode
        s_a = self.gen_b.encode(self.single(x_a))
        s_b = self.gen_b.encode(self.single(x_b))
        f_a, p_a, fe_a = self.id_b(scale2(x_a))
        f_b, p_b, fe_b = self.id_b(scale2(x_b))
        # autodecode
        x_a_recon = self.gen_b.decode(s_a, f_a)
        x_b_recon = self.gen_b.decode(s_b, f_b)

        # encode the same ID different photo
        fp_a, pp_a, fe_pa = self.id_b(scale2(xp_a))
        fp_b, pp_b, fe_pb = self.id_b(scale2(xp_b))

        # decode the same person
        x_a_recon_p = self.gen_b.decode(s_a, fp_a)
        x_b_recon_p = self.gen_b.decode(s_b, fp_b)

        # has gradient
        x_ba = self.gen_b.decode(s_b, f_a)
        x_ab = self.gen_b.decode(s_a, f_b)
        # no gradient
        x_ba_copy = Variable(x_ba.data, requires_grad=False)
        x_ab_copy = Variable(x_ab.data, requires_grad=False)

        rand_num = random.uniform(0, 1)
        #################################
        # encode structure
        if hyperparameters['use_encoder_again'] >= rand_num:
            # encode again (encoder is tuned, input is fixed)
            s_a_recon = self.gen_b.enc_content(self.single(x_ab_copy))
            s_b_recon = self.gen_b.enc_content(self.single(x_ba_copy))
        else:
            # copy the encoder
            self.enc_content_copy = copy.deepcopy(self.gen_b.enc_content)
            self.enc_content_copy = self.enc_content_copy.eval()
            # encode again (encoder is fixed, input is tuned)
            s_a_recon = self.enc_content_copy(self.single(x_ab))
            s_b_recon = self.enc_content_copy(self.single(x_ba))

        #################################
        # encode appearance
        self.id_a_copy = copy.deepcopy(self.id_b)
        self.id_a_copy = self.id_a_copy.eval()
        if hyperparameters['train_bn']:
            self.id_a_copy = self.id_a_copy.apply(train_bn)
        self.id_b_copy = self.id_a_copy
        # encode again (encoder is fixed, input is tuned)
        f_a_recon, p_a_recon, _ = self.id_a_copy(scale2(x_ba))
        f_b_recon, p_b_recon, _ = self.id_b_copy(scale2(x_ab))

        # teacher Loss
        self.loss_teacher = 0.0

        # decode again (if needed)
        if hyperparameters['use_decoder_again']:
            x_aba = self.gen_b.decode(s_a_recon, f_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
            x_bab = self.gen_b.decode(s_b_recon, f_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        else:
            self.mlp_w_copy = copy.deepcopy(self.gen_b.mlp_w)
            self.mlp_b_copy = copy.deepcopy(self.gen_b.mlp_b)
            self.dec_copy = copy.deepcopy(self.gen_b.dec)  # Error
            ID = f_a_recon
            ID_Style = ID.view(ID.shape[0], ID.shape[1], 1, 1)
            adain_params_w = self.mlp_w_copy(ID_Style)
            adain_params_b = self.mlp_b_copy(ID_Style)
            self.gen_b.assign_adain_params(adain_params_w, adain_params_b, self.dec_copy)
            x_aba = self.dec_copy(s_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

            ID = f_b_recon
            ID_Style = ID.view(ID.shape[0], ID.shape[1], 1, 1)
            adain_params_w = self.mlp_w_copy(ID_Style)
            adain_params_b = self.mlp_b_copy(ID_Style)
            self.gen_b.assign_adain_params(adain_params_w, adain_params_b, self.dec_copy)
            x_bab = self.dec_copy(s_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # auto-encoder image reconstruction
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_xp_a = self.recon_criterion(x_a_recon_p, x_a)
        self.loss_gen_recon_xp_b = self.recon_criterion(x_b_recon_p, x_b)

        # feature reconstruction
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b) if hyperparameters['recon_s_w'] > 0 else 0
        self.loss_gen_recon_f_a = self.recon_criterion(f_a_recon, f_a) if hyperparameters['recon_f_w'] > 0 else 0
        self.loss_gen_recon_f_b = self.recon_criterion(f_b_recon, f_b) if hyperparameters['recon_f_w'] > 0 else 0

        # Random Erasing only effect the ID and PID loss.
        if hyperparameters['erasing_p'] > 0:
            x_a_re = self.to_re(scale2(x_a.clone()))
            x_b_re = self.to_re(scale2(x_b.clone()))
            xp_a_re = self.to_re(scale2(xp_a.clone()))
            xp_b_re = self.to_re(scale2(xp_b.clone()))
            _, p_a, _ = self.id_b(x_a_re)
            _, p_b, _ = self.id_b(x_b_re)
            # encode the same ID different photo
            _, pp_a, _ = self.id_b(xp_a_re)
            _, pp_b, _ = self.id_b(xp_b_re)

        # ID loss AND Tune the Generated image
        if hyperparameters['id_tgt']:
            weight_B = hyperparameters['teacher_w'] * hyperparameters['B_w']
            self.loss_id = self.id_criterion(p_a[0], l_a) + self.id_criterion(p_b[0], l_b) \
                           + weight_B * (self.id_criterion(p_a[1], l_a) + self.id_criterion(p_b[1], l_b))
            self.loss_pid = self.id_criterion(pp_a[0], l_a) + self.id_criterion(pp_b[0],
                                                                                l_b)  # + weight_B * ( self.id_criterion(pp_a[1], l_a) + self.id_criterion(pp_b[1], l_b) )
            self.loss_pid *= self.loss_pid*hyperparameters['tgt_pos']
            self.loss_gen_recon_id = self.id_criterion(p_a_recon[0], l_a) + self.id_criterion(p_b_recon[0], l_b)
        else:
            self.loss_id = 0.0
            self.loss_pid = 0.0
            self.loss_gen_recon_id = 0.0

        # print(f_a_recon, f_a)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_a.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0

        # ID domain adversarial loss
        self.loss_gen_id_adv = ( self.id_dis.calc_gen_loss(fe_b) + self.id_dis.calc_gen_loss(fe_a) ) / 2 if hyperparameters['id_adv_w'] > 0 else 0

        if iteration > hyperparameters['warm_iter']:
            hyperparameters['recon_f_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_f_w'] = min(hyperparameters['recon_f_w'], hyperparameters['max_w'])
            hyperparameters['recon_s_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_s_w'] = min(hyperparameters['recon_s_w'], hyperparameters['max_w'])
            hyperparameters['recon_x_cyc_w'] += hyperparameters['warm_scale']
            hyperparameters['recon_x_cyc_w'] = min(hyperparameters['recon_x_cyc_w'], hyperparameters['max_cyc_w'])

        if iteration > hyperparameters['warm_teacher_iter']:
            hyperparameters['teacher_w'] += hyperparameters['warm_scale']
            hyperparameters['teacher_w'] = min(hyperparameters['teacher_w'], hyperparameters['max_teacher_w'])

        hyperparameters['id_adv_w'] += hyperparameters['adv_warm_scale']
        hyperparameters['id_adv_w'] = min(hyperparameters['id_adv_w'], hyperparameters['id_adv_w_max'])
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_xp_w'] * hyperparameters['recon_xp_tgt_w'] * self.loss_gen_recon_xp_a + \
                              hyperparameters['recon_f_w'] * self.loss_gen_recon_f_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_xp_w'] * hyperparameters['recon_xp_tgt_w'] * self.loss_gen_recon_xp_b + \
                              hyperparameters['recon_f_w'] * self.loss_gen_recon_f_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['id_w'] * self.loss_id + \
                              hyperparameters['pid_w'] * self.loss_pid + \
                              hyperparameters['recon_id_w'] * self.loss_gen_recon_id + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
                              hyperparameters['teacher_w'] * self.loss_teacher + \
                              hyperparameters['id_adv_w'] * self.loss_gen_id_adv
        self.loss_gen_total.backward()
        self.gen_b_opt.step()
        self.id_opt.step()
        print(
            "L_total: %.4f, L_gan: %.4f, L_adv: %.4f, Lx: %.4f, Lxp: %.4f, Lrecycle:%.4f, Lf: %.4f, Ls: %.4f, Recon-id: %.4f, id: %.4f, pid:%.4f, teacher: %.4f" % (
            self.loss_gen_total, \
            hyperparameters['gan_w'] * (self.loss_gen_adv_a + self.loss_gen_adv_b), \
            hyperparameters['id_adv_w'] * (self.loss_gen_id_adv), \
            hyperparameters['recon_x_w'] * (self.loss_gen_recon_x_a + self.loss_gen_recon_x_b), \
            hyperparameters['recon_xp_w'] * hyperparameters['recon_xp_tgt_w'] * (self.loss_gen_recon_xp_a + self.loss_gen_recon_xp_b), \
            hyperparameters['recon_x_cyc_w'] * (self.loss_gen_cycrecon_x_a + self.loss_gen_cycrecon_x_b), \
            hyperparameters['recon_f_w'] * (self.loss_gen_recon_f_a + self.loss_gen_recon_f_b), \
            hyperparameters['recon_s_w'] * (self.loss_gen_recon_s_a + self.loss_gen_recon_s_b), \
            hyperparameters['recon_id_w'] * self.loss_gen_recon_id, \
            hyperparameters['id_w'] * self.loss_id, \
            hyperparameters['pid_w'] * self.loss_pid, \
            hyperparameters['teacher_w'] * self.loss_teacher))

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample_ab(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2, x_aba, x_bab = [], [], [], [], [], [], [], []
        for i in range(x_a.size(0)):
            s_a = self.gen_a.encode(self.single(x_a[i].unsqueeze(0)))
            s_b = self.gen_b.encode(self.single(x_b[i].unsqueeze(0)))
            f_a, _, _ = self.id_a(scale2(x_a[i].unsqueeze(0)))
            f_b, _, _ = self.id_b(scale2(x_b[i].unsqueeze(0)))
            x_a_recon.append(self.gen_a.decode(s_a, f_a))
            x_b_recon.append(self.gen_b.decode(s_b, f_b))
            x_ba = self.gen_b.decode(s_b, f_a)
            x_ab = self.gen_a.decode(s_a, f_b)
            x_ba1.append(x_ba)
            x_ba2.append(self.gen_b.decode(s_b, f_a))
            x_ab1.append(x_ab)
            x_ab2.append(self.gen_a.decode(s_a, f_b))
            # cycle
            s_b_recon = self.gen_b.enc_content(self.single(x_ba))
            s_a_recon = self.gen_a.enc_content(self.single(x_ab))
            f_a_recon, _, _ = self.id_a(scale2(x_ba))
            f_b_recon, _, _ = self.id_b(scale2(x_ab))
            x_aba.append(self.gen_a.decode(s_a_recon, f_a_recon))
            x_bab.append(self.gen_b.decode(s_b_recon, f_b_recon))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_aba, x_bab = torch.cat(x_aba), torch.cat(x_bab)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()

        return x_a, x_a_recon, x_aba, x_ab1, x_ab2, x_b, x_b_recon, x_bab, x_ba1, x_ba2

    def sample_aa(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2, x_aba, x_bab = [], [], [], [], [], [], [], []
        for i in range(x_a.size(0)):
            s_a = self.gen_a.encode(self.single(x_a[i].unsqueeze(0)))
            s_b = self.gen_a.encode(self.single(x_b[i].unsqueeze(0)))
            f_a, _, _ = self.id_a(scale2(x_a[i].unsqueeze(0)))
            f_b, _, _ = self.id_a(scale2(x_b[i].unsqueeze(0)))
            x_a_recon.append(self.gen_a.decode(s_a, f_a))
            x_b_recon.append(self.gen_a.decode(s_b, f_b))
            x_ba = self.gen_a.decode(s_b, f_a)
            x_ab = self.gen_a.decode(s_a, f_b)
            x_ba1.append(x_ba)
            x_ba2.append(self.gen_a.decode(s_b, f_a))
            x_ab1.append(x_ab)
            x_ab2.append(self.gen_a.decode(s_a, f_b))
            # cycle
            s_b_recon = self.gen_a.enc_content(self.single(x_ba))
            s_a_recon = self.gen_a.enc_content(self.single(x_ab))
            f_a_recon, _, _ = self.id_a(scale2(x_ba))
            f_b_recon, _, _ = self.id_a(scale2(x_ab))
            x_aba.append(self.gen_a.decode(s_a_recon, f_a_recon))
            x_bab.append(self.gen_a.decode(s_b_recon, f_b_recon))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_aba, x_bab = torch.cat(x_aba), torch.cat(x_bab)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()

        return x_a, x_a_recon, x_aba, x_ab1, x_ab2, x_b, x_b_recon, x_bab, x_ba1, x_ba2

    def sample_bb(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2, x_aba, x_bab = [], [], [], [], [], [], [], []
        for i in range(x_a.size(0)):
            s_a = self.gen_b.encode(self.single(x_a[i].unsqueeze(0)))
            s_b = self.gen_b.encode(self.single(x_b[i].unsqueeze(0)))
            f_a, _, _ = self.id_b(scale2(x_a[i].unsqueeze(0)))
            f_b, _, _ = self.id_b(scale2(x_b[i].unsqueeze(0)))
            x_a_recon.append(self.gen_b.decode(s_a, f_a))
            x_b_recon.append(self.gen_b.decode(s_b, f_b))
            x_ba = self.gen_b.decode(s_b, f_a)
            x_ab = self.gen_b.decode(s_a, f_b)
            x_ba1.append(x_ba)
            x_ba2.append(self.gen_b.decode(s_b, f_a))
            x_ab1.append(x_ab)
            x_ab2.append(self.gen_b.decode(s_a, f_b))
            # cycle
            s_b_recon = self.gen_b.enc_content(self.single(x_ba))
            s_a_recon = self.gen_b.enc_content(self.single(x_ab))
            f_a_recon, _, _ = self.id_b(scale2(x_ba))
            f_b_recon, _, _ = self.id_b(scale2(x_ab))
            x_aba.append(self.gen_b.decode(s_a_recon, f_a_recon))
            x_bab.append(self.gen_b.decode(s_b_recon, f_b_recon))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_aba, x_bab = torch.cat(x_aba), torch.cat(x_bab)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()

        return x_a, x_a_recon, x_aba, x_ab1, x_ab2, x_b, x_b_recon, x_bab, x_ba1, x_ba2

    def dis_update_ab(self, x_a, x_b, hyperparameters):
        self.dis_a_opt.zero_grad()
        self.id_dis_opt.zero_grad()
        # self.dis_b_opt.zero_grad()
        # encode
        # x_a_single = self.single(x_a)
        s_a = self.gen_a.encode(self.single(x_a))
        s_b = self.gen_b.encode(self.single(x_b))
        f_a, _, fe_a = self.id_a(scale2(x_a))
        f_b, _, fe_b = self.id_b(scale2(x_b))
        # decode (cross domain)
        x_ba = self.gen_b.decode(s_b, f_a)
        x_ab = self.gen_a.decode(s_a, f_b)
        # print(x_ab)
        # D loss
        self.loss_dis_a, reg_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b, reg_b = self.dis_a.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_id_dis_ab, _, _ = self.id_dis.calc_dis_loss_ab(fe_a.detach(), fe_b.detach())
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_id_dis_total = hyperparameters['id_adv_w'] * self.loss_id_dis_ab
        print("DLoss: %.4f" % self.loss_dis_total, "Reg: %.4f" % (reg_a + reg_b), "ID_adv: %.4f" % self.loss_id_dis_total)
        self.loss_dis_total.backward()
        self.loss_id_dis_total.backward()
        # check gradient norm
        self.loss_total_norm = 0.0
        for p in self.id_dis.parameters():
            param_norm = p.grad.data.norm(2)
            self.loss_total_norm += param_norm.item() ** 2
        self.loss_total_norm =  self.loss_total_norm ** (1. / 2)
        #
        self.dis_a_opt.step()
        self.id_dis_opt.step()
        # self.dis_b_opt.step()

    def dis_update_aa(self, x_a, x_b, hyperparameters):
        self.dis_a_opt.zero_grad()
        self.id_dis_opt.zero_grad()
        # encode
        # x_a_single = self.single(x_a)
        s_a = self.gen_a.encode(self.single(x_a))
        s_b = self.gen_a.encode(self.single(x_b))
        f_a, _, fe_a = self.id_a(scale2(x_a))
        f_b, _, fe_b = self.id_a(scale2(x_b))
        # decode (cross domain)
        x_ba = self.gen_a.decode(s_b, f_a)
        x_ab = self.gen_a.decode(s_a, f_b)
        # print(x_ab)
        # D loss
        self.loss_dis_a, reg_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b, reg_b = self.dis_a.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_id_dis_aa, _, _ = self.id_dis.calc_dis_loss_aa(fe_a.detach(), fe_b.detach())
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_id_dis_total = hyperparameters['id_adv_w'] * self.loss_id_dis_aa
        print("DLoss: %.4f" % self.loss_dis_total, "Reg: %.4f" % (reg_a + reg_b), "ID_adv: %.4f" % self.loss_id_dis_total)
        self.loss_dis_total.backward()
        self.loss_id_dis_total.backward()
        # check gradient norm
        self.loss_total_norm = 0.0
        for p in self.id_dis.parameters():
            param_norm = p.grad.data.norm(2)
            self.loss_total_norm += param_norm.item() ** 2
        self.loss_total_norm =  self.loss_total_norm ** (1. / 2)
        #
        self.dis_a_opt.step()
        self.id_dis_opt.step()

    def dis_update_bb(self, x_a, x_b, hyperparameters):
        self.dis_a_opt.zero_grad()
        self.id_dis_opt.zero_grad()
        # encode
        # x_a_single = self.single(x_a)
        s_a = self.gen_b.encode(self.single(x_a))
        s_b = self.gen_b.encode(self.single(x_b))
        f_a, _, fe_a = self.id_b(scale2(x_a))
        f_b, _, fe_b = self.id_b(scale2(x_b))
        # decode (cross domain)
        x_ba = self.gen_b.decode(s_b, f_a)
        x_ab = self.gen_b.decode(s_a, f_b)
        # print(x_ab)
        # D loss
        self.loss_dis_a, reg_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b, reg_b = self.dis_a.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_id_dis_bb, _, _ = self.id_dis.calc_dis_loss_bb(fe_a.detach(), fe_b.detach())
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_id_dis_total = hyperparameters['id_adv_w'] * self.loss_id_dis_bb
        print("DLoss: %.4f" % self.loss_dis_total, "Reg: %.4f" % (reg_a + reg_b), "ID_adv: %.4f" % self.loss_id_dis_total)
        self.loss_dis_total.backward()
        self.loss_id_dis_total.backward()
        # check gradient norm
        self.loss_total_norm = 0.0
        for p in self.id_dis.parameters():
            param_norm = p.grad.data.norm(2)
            self.loss_total_norm += param_norm.item() ** 2
        self.loss_total_norm =  self.loss_total_norm ** (1. / 2)
        #
        self.dis_a_opt.step()
        self.id_dis_opt.step()

    def update_learning_rate(self):
        if self.dis_a_scheduler is not None:
            self.dis_a_scheduler.step()
        # if self.dis_b_scheduler is not None:
        #     self.dis_b_scheduler.step()
        if self.gen_a_scheduler is not None:
            self.gen_a_scheduler.step()
        if self.gen_b_scheduler is not None:
            self.gen_b_scheduler.step()
        if self.id_scheduler is not None:
            self.id_scheduler.step()
        if self.id_dis_scheduler is not None:
            self.id_dis_scheduler.step()

    def scale_learning_rate(self, lr_decayed, lr_recover, hyperparameters):
        if not lr_decayed:
            if lr_recover:
                for g in self.dis_a_opt.param_groups:
                    g['lr'] *= hyperparameters['gamma']
                for g in self.gen_a_opt.param_groups:
                    g['lr'] *= hyperparameters['gamma']
                for g in self.gen_b_opt.param_groups:
                    g['lr'] *= hyperparameters['gamma']
                for g in self.id_opt.param_groups:
                    g['lr'] *= hyperparameters['gamma2']
                for g in self.id_dis_opt.param_groups:
                    g['lr'] *= hyperparameters['gamma2']
            elif not lr_recover:
                for g in self.id_opt.param_groups:
                    g['lr'] = g['lr'] * hyperparameters['lr2_ramp_factor']
        elif lr_decayed:
            for g in self.dis_a_opt.param_groups:
                g['lr'] = g['lr'] / hyperparameters['gamma'] * hyperparameters['lr2_ramp_factor']
            for g in self.gen_a_opt.param_groups:
                g['lr'] = g['lr'] / hyperparameters['gamma'] * hyperparameters['lr2_ramp_factor']
            for g in self.gen_b_opt.param_groups:
                g['lr'] = g['lr'] / hyperparameters['gamma'] * hyperparameters['lr2_ramp_factor']
            for g in self.id_opt.param_groups:
                g['lr'] = g['lr'] / hyperparameters['gamma2'] * hyperparameters['lr2_ramp_factor']
            for g in self.id_dis_opt.param_groups:
                g['lr'] = g['lr'] / hyperparameters['gamma2'] * hyperparameters['lr2_ramp_factor']

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen_a")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        last_model_name = get_model_list(checkpoint_dir, "gen_b")
        state_dict = torch.load(last_model_name)
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis_a")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        # last_model_name = get_model_list(checkpoint_dir, "dis_b")
        # state_dict = torch.load(last_model_name)
        self.dis_b = self.dis_a
        # Load ID dis
        last_model_name = get_model_list(checkpoint_dir, "id")
        state_dict = torch.load(last_model_name)
        self.id_a.load_state_dict(state_dict['a'])
        self.id_b = self.id_a
        # Load optimizers
        try:
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.dis_a_opt.load_state_dict(state_dict['dis_a'])
            # self.dis_b_opt.load_state_dict(state_dict['dis_b'])
            self.gen_a_opt.load_state_dict(state_dict['gen_a'])
            self.gen_b_opt.load_state_dict(state_dict['gen_b'])
            self.id_opt.load_state_dict(state_dict['id'])
        except:
            pass
        # Reinitilize schedulers
        self.dis_a_scheduler = get_scheduler(self.dis_a_opt, hyperparameters, iterations)
        # self.dis_b_scheduler = get_scheduler(self.dis_b_opt, hyperparameters, iterations)
        self.gen_a_scheduler = get_scheduler(self.gen_a_opt, hyperparameters, iterations)
        self.gen_b_scheduler = get_scheduler(self.gen_b_opt, hyperparameters, iterations)
        self.id_scheduler = get_scheduler(self.id_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def resume_DAt0(self, checkpoint_dir):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'],strict=False)
        # last_model_name = get_model_list(checkpoint_dir, "gen_b")
        # state_dict = torch.load(last_model_name)
        self.gen_b.load_state_dict(state_dict['a'],strict=False)
        iterations = 0
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'],strict=False)
        # last_model_name = get_model_list(checkpoint_dir, "dis_b")
        # state_dict = torch.load(last_model_name)
        self.dis_b = self.dis_a
        # Load ID dis
        last_model_name = get_model_list(checkpoint_dir, "id")
        state_dict = torch.load(last_model_name)
        classifier1 = self.id_a.classifier1.classifier
        classifier2 = self.id_a.classifier2.classifier
        self.id_a.classifier1.classifier = nn.Sequential()
        self.id_a.classifier2.classifier = nn.Sequential()
        self.id_a.load_state_dict(state_dict['a'], strict=False)
        self.id_a.classifier1.classifier = classifier1
        self.id_a.classifier2.classifier = classifier2
        self.id_b = self.id_a
        print('Resume from iteration %d' % iterations)
        #self.save(checkpoint_dir, 0)
        return iterations

    def resume_DAt1(self, checkpoint_dir):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen_a")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        # last_model_name = get_model_list(checkpoint_dir, "gen_b")
        # state_dict = torch.load(last_model_name)
        last_model_name = get_model_list(checkpoint_dir, "gen_b")
        state_dict = torch.load(last_model_name)
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = 0
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis_a")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        # last_model_name = get_model_list(checkpoint_dir, "dis_b")
        # state_dict = torch.load(last_model_name)
        self.dis_b = self.dis_a
        # Load ID dis
        last_model_name = get_model_list(checkpoint_dir, "id")
        state_dict = torch.load(last_model_name)
        classifier1 = self.id_a.classifier1.classifier
        classifier2 = self.id_a.classifier2.classifier
        self.id_a.classifier1.classifier = nn.Sequential()
        self.id_a.classifier2.classifier = nn.Sequential()
        self.id_a.load_state_dict(state_dict['a'], strict=False)
        self.id_a.classifier1.classifier = classifier1
        self.id_a.classifier2.classifier = classifier2
        self.id_b = self.id_a
        print('Resume from iteration %d' % iterations)
        #self.save(checkpoint_dir, 0)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_a_name = os.path.join(snapshot_dir, 'gen_a_%08d.pt' % (iterations + 1))
        gen_b_name = os.path.join(snapshot_dir, 'gen_b_%08d.pt' % (iterations + 1))
        dis_a_name = os.path.join(snapshot_dir, 'dis_a_%08d.pt' % (iterations + 1))
        dis_b_name = os.path.join(snapshot_dir, 'dis_b_%08d.pt' % (iterations + 1))
        id_name = os.path.join(snapshot_dir, 'id_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict()}, gen_a_name)
        torch.save({'b': self.gen_b.state_dict()}, gen_b_name)
        torch.save({'a': self.dis_a.state_dict()}, dis_a_name)
        torch.save({'b': self.dis_b.state_dict()}, dis_b_name)
        torch.save({'a': self.id_a.state_dict()}, id_name)
        torch.save({'gen_a': self.gen_a_opt.state_dict(), 'gen_b': self.gen_b_opt.state_dict(), 'id': self.id_opt.state_dict(), 'dis_a': self.dis_a_opt.state_dict(), 'dis_b': self.dis_a_opt.state_dict()},
                   opt_name)

    def test(self, opt):
        self.eval()
        test_dir = opt['data_root_b']
        data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 128), interpolation=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ############### Ten Crop
            # transforms.TenCrop(224),
            # transforms.Lambda(lambda crops: torch.stack(
            #   [transforms.ToTensor()(crop)
            #      for crop in crops]
            # )),
            # transforms.Lambda(lambda crops: torch.stack(
            #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
            #       for crop in crops]
            # ))
        ])
        data_dir = test_dir
        image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                          ['gallery', 'query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt['test_batchsize'],
                                                      shuffle=False, num_workers=0) for x in ['gallery', 'query']}

        gallery_path = image_datasets['gallery'].imgs
        query_path = image_datasets['query'].imgs

        gallery_cam, gallery_label, _ = get_id(gallery_path, time_constraint = opt['time_constraint'])
        query_cam, query_label, _ = get_id(query_path, time_constraint = opt['time_constraint'])


        ######################################################################
        # Load Collected data Trained model
        print('-------test-----------')

        # Extract feature
        with torch.no_grad():
            gallery_feature = self.extract_feature(dataloaders['gallery'], opt)
            query_feature = self.extract_feature(dataloaders['query'], opt)

        gallery_label = np.array(gallery_label)
        gallery_cam = np.array(gallery_cam)
        query_label = np.array(query_label)
        query_cam = np.array(query_cam)
        alpha = [0, 0.5, -1]
        mAP_alpha = [0]*3
        # print(query_label)
        for j in range(len(alpha)):
            CMC = torch.IntTensor(len(gallery_label)).zero_()
            ap = 0.0
            for i in range(len(query_label)):
                qf = query_feature[i].clone()
                if alpha[j] == -1:
                    qf[0:512] *= 0
                else:
                    qf[512:1024] *= alpha[j]

                ap_tmp, CMC_tmp = evaluate(qf, query_label[i], query_cam[i], gallery_feature, gallery_label,
                                           gallery_cam)
                if CMC_tmp[0] == -1:
                    continue
                CMC = CMC + CMC_tmp
                ap += ap_tmp
                # print(i, CMC_tmp[0])

            CMC = CMC.float()
            CMC = CMC / len(query_label)  # average CMC
            print('Alpha:%.2f Rank@1:%.4f Rank@5:%.4f Rank@10:%.4f mAP:%.4f' % (
            alpha[j], CMC[0], CMC[4], CMC[9], ap / len(query_label)))
            mAP_alpha[j] = ap / len(query_label)
        self.rank_1 = CMC[0]
        self.rank_5 = CMC[4]
        self.rank_10 = CMC[9]
        self.mAP_zero = mAP_alpha[0]
        self.mAP_half = mAP_alpha[1]
        self.mAP_neg_one = mAP_alpha[2]

        del gallery_feature, query_feature, query_label
        self.train()

        return

    def pseudo_label_generate(self, opt):
        ### Feature extraction ###
        self.eval()
        test_dir = opt['data_root_b']
        data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 128), interpolation=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        data_dir = test_dir
        image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['train_all']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt['test_batchsize'],
                                                      shuffle=False, num_workers=0) for x in ['train_all']}
        train_path = image_datasets['train_all'].imgs
        # Extract feature
        with torch.no_grad():
            train_feature = self.extract_feature(dataloaders['train_all'], opt)
        self.train()

        ### clustering ###
        labels = self.clustering(train_feature, train_path, opt)

        ### copy and save images ###
        n_samples = train_feature.shape[0]
        opt['ID_class_b'] = int(max(labels)) + 1
        self.copy_save(labels, train_path, n_samples, opt)
        return

    def clustering(self, train_feature, train_path, opt):

        ######################################################################
        alpha = 0.5

        n_samples = train_feature.shape[0]
        train_feature_clone = train_feature.clone()
        train_feature_clone[:, 512:1024] *= alpha  # since we count 0.5 for the fine-grained feature. 0.7*0.7=0.49
        train_dist = torch.mm(train_feature_clone, torch.transpose(train_feature, 0, 1)) / (1 + alpha)
        print(train_dist)

        if opt['time_constraint']:
            print('--------------------------Use Time Constraint---------------------------')
            train_camera_id, train_time_id, train_labels = get_id(train_path, time_constraint = opt['time_constraint'])
            train_time_id = np.asarray(train_time_id)
            train_camera_id = np.asarray(train_camera_id)

            # Long Time
            for i in range(n_samples):
                t_time = train_time_id[i]
                index = np.argwhere(np.absolute(train_time_id - t_time) > 40000).flatten()
                train_dist[i, index] = -1
                print(len(index))

            # Same Camera Long Time
            for i in range(n_samples):
                t_time = train_time_id[i]
                t_cam = train_camera_id[i]
                index = np.argwhere(np.absolute(train_time_id - t_time) > 5000).flatten()
                c_index = np.argwhere(train_camera_id == t_cam).flatten()
                index = np.intersect1d(index, c_index)
                train_dist[i, index] = -1
                print(len(index))

        print('--------------------------Start Re-ranking---------------------------')
        train_dist = re_ranking_one(train_dist.cpu().numpy())
        print('--------------------------Clustering---------------------------')
        # cluster
        min_samples = opt['clustering']['min_samples']
        eps = opt['clustering']['eps']

        cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=8)
        ### non-negative clustering
        train_dist = np.maximum(train_dist, 0)
        ###
        cluster = cluster.fit(train_dist)
        print('Cluster Class Number:  %d' % len(np.unique(cluster.labels_)))
        # center = cluster.core_sample_indices_
        labels = cluster.labels_

        return labels

    def copy_save(self, labels, train_path, n_samples, opt):
        ### copy pseudo-labels in target ###
        save_path = opt['data_root'] + '/train_all'
        sample_b_valid = 0
        for i in range(n_samples):
            if labels[i] != -1:
                src_path = train_path[i][0]
                dst_id = labels[i]
                dst_path = save_path + '/' + 'B_' + str(int(dst_id))
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + os.path.basename(src_path))
                sample_b_valid += 1

        opt['sample_b'] = sample_b_valid

        ### copy ground truth in source ###
        # train_all
        src_all_path = opt['data_root_a']
        # for dukemtmc-reid, we do not need multi-query
        src_train_all_path = os.path.join(src_all_path, 'train_all')
        subfolder_list = os.listdir(src_train_all_path)
        file_list = []
        for path, subdirs, files in os.walk(src_train_all_path):
            for name in files:
                file_list.append(os.path.join(path, name))
        opt['ID_class_a'] = len(subfolder_list)
        opt['sample_a'] = len(file_list)
        for name in subfolder_list:
            copytree(src_train_all_path + '/' + name, save_path + '/A_' + name)

        return


    def extract_feature(self, dataloaders, opt):
        model = copy.deepcopy(self.id_a)
        if opt['train_bn']:
            model = model.apply(train_bn)
        # Remove the final fc layer and classifier layer
        model.model.fc = nn.Sequential()
        model.classifier1.classifier = nn.Sequential()
        model.classifier2.classifier = nn.Sequential()
        model.eval()
        features = torch.FloatTensor()
        count = 0
        for data in dataloaders:
            img, label = data
            img, label = img.cuda().detach(), label.cuda().detach()
            n, c, h, w = img.size()
            count += n
            #print(count)
            ff = torch.FloatTensor(n,1024).zero_()
            for i in range(2):
                if(i==1):
                    img = fliplr(img)
                input_img = Variable(img)
                f, x, _ = model(input_img)
                x[0] = norm(x[0])
                x[1] = norm(x[1])
                f = torch.cat((x[0],x[1]), dim=1) #use 512-dim feature
                f = f.data.cpu()
                ff = ff+f

            ff[:, 0:512] = norm(ff[:, 0:512], dim=1)
            ff[:, 512:1024] = norm(ff[:, 512:1024], dim =1)
            features = torch.cat((features,ff), 0)
        del model
        return features

