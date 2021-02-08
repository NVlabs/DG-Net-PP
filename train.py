"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_mix_data_loaders, get_data_loader_folder, prepare_sub_folder_pseudo, write_html, write_loss, get_config, write_2images, Timer
import argparse
from trainer import DGNetpp_Trainer
import torch.backends.cudnn as cudnn
import torch
import random as rn
import numpy.random as random
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

# set random seed
def set_seed(seed=0):
   rn.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
   # cudnn.enabled = False
   cudnn.deterministic = True
   cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/latest.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--name', type=str, default='latest_ablation', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='DGNet++', help="DGNet++")
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
opts = parser.parse_args()

str_ids = opts.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gpu_ids.append(int(str_id))
num_gpu = len(gpu_ids)
if num_gpu > 1:
    raise Exception('Currently only single GPU training is supported!')

# Load experiment setting
config = get_config(opts.config)
set_seed(config['randseed'])
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# preparing sampling images
train_loader_a_sample = get_data_loader_folder(os.path.join(config['data_root_a'], 'train_all'), config['batch_size'], False,
                                        config['new_size'], config['crop_image_height'], config['crop_image_width'], config['num_workers'], False)
train_loader_b_sample = get_data_loader_folder(os.path.join(config['data_root_b'], 'train_all'), config['batch_size'], False,
                                             config['new_size'], config['crop_image_height'], config['crop_image_width'], config['num_workers'], False)

train_aba_rand = random.permutation(train_loader_a_sample.dataset.img_num)[0:display_size]
train_abb_rand = random.permutation(train_loader_b_sample.dataset.img_num)[0:display_size]
train_aab_rand = random.permutation(train_loader_a_sample.dataset.img_num)[0:display_size]
train_bbb_rand = random.permutation(train_loader_b_sample.dataset.img_num)[0:display_size]

train_display_images_aba = torch.stack([train_loader_a_sample.dataset[i][0] for i in train_aba_rand]).cuda()
train_display_images_abb = torch.stack([train_loader_b_sample.dataset[i][0] for i in train_abb_rand]).cuda()
train_display_images_aaa = torch.stack([train_loader_a_sample.dataset[i][0] for i in train_aba_rand]).cuda()
train_display_images_aab = torch.stack([train_loader_a_sample.dataset[i][0] for i in train_aab_rand]).cuda()
train_display_images_bba = torch.stack([train_loader_b_sample.dataset[i][0] for i in train_abb_rand]).cuda()
train_display_images_bbb = torch.stack([train_loader_b_sample.dataset[i][0] for i in train_bbb_rand]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
else:
    shutil.rmtree(output_directory)
    os.makedirs(output_directory)
shutil.copyfile(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder
shutil.copyfile('trainer.py', os.path.join(output_directory, 'trainer.py')) # copy file to output folder
shutil.copyfile('reIDmodel.py', os.path.join(output_directory, 'reIDmodel.py')) # copy file to output folder
shutil.copyfile('networks.py', os.path.join(output_directory, 'networks.py')) # copy file to output folder

checkpoint_directory_prev = config['src_model_dir']

countaa, countab, countba, countbb = 1, 1, 1, 1
count_dis_update = config['dis_update_iter']
nepoch = 0
iterations = 0
epoch_round = config['epoch_round_adv']
lr_decayed = False
mAP_list, rank1_list, rank5_list, rank10_list = [], [], [], []
for round_idx in range(config['max_round']):
    ### setup folders
    round_output_directory = os.path.join(output_directory, str(round_idx))
    checkpoint_directory, image_directory, pseudo_directory = prepare_sub_folder_pseudo(round_output_directory)
    config['data_root'] = pseudo_directory

    # In the initial round, we disenable self-training and warmup the network with adversarial training
    # At the round of adv_warm_max_round, we switch to self-training
    if round_idx == config['adv_warm_max_round']:
        config['lr2'] *= config['lr2_ramp_factor']
        config['id_adv_w'] = 0.0
        config['id_adv_w_max'] = 0.0
        config['id_tgt'] = True
        config['teacher'] = '' # we do not use teacher in the self-training
        if config['aa_drop']:
            config['aa'] = False

    ### Evaluate source model ###
    if round_idx == 0:
        ### Model initialization with source model for test ###
        if opts.trainer == 'DGNet++':
            trainer = DGNetpp_Trainer(config)
        trainer.cuda()
        _ = trainer.resume_DAt1(checkpoint_directory_prev) if round_idx > 0 else trainer.resume_DAt0(checkpoint_directory_prev)

        trainer.test(config)
        write_loss(iterations, trainer, train_writer)
        rank1 = trainer.rank_1
        rank5 = trainer.rank_5
        rank10 = trainer.rank_10
        mAP0 = trainer.mAP_zero
        mAP05 = trainer.mAP_half
        mAPn1 = trainer.mAP_neg_one

        mAP_list.append(mAP05)
        rank1_list.append(rank1.numpy())
        rank5_list.append(rank5.numpy())
        rank10_list.append(rank10.numpy())

    ### Pseudo-label generation ###
    trainer.pseudo_label_generate(config)

    ### Model initialization w.r.t. current pseudo labels for train ###
    if round_idx == 0:
        config['ID_class_b'] = 0 # In the initial round, we disenable self-training
    if opts.trainer == 'DGNet++':
        trainer = DGNetpp_Trainer(config)
    trainer.cuda()
    _ = trainer.resume_DAt1(checkpoint_directory_prev) if round_idx > 0 else trainer.resume_DAt0(checkpoint_directory_prev)

    trainer.rank_1 = rank1
    trainer.rank_5 = rank5
    trainer.rank_10 = rank10
    trainer.mAP_zero = mAP0
    trainer.mAP_half = mAP05
    trainer.mAP_neg_one = mAPn1
    ### DGNet++ Training ###
    # data initialize
    train_loader_a, train_loader_b, _, _ = get_mix_data_loaders(config)
    print('Note that dataloader may hang with too much nworkers.')
    mixData_size = 2 * min(config['sample_a'], config['sample_b'])
    config['epoch_iteration'] = mixData_size // config['batch_size']
    print('Every epoch need %d iterations' % config['epoch_iteration'])

    # training
    subiterations = 0
    epoch_ridx = 0
    while epoch_ridx < epoch_round:
        for it, ((images_a, labels_a, pos_a), (images_b, labels_b, pos_b)) in enumerate(zip(train_loader_a, train_loader_b)):
            trainer.update_learning_rate()

            print('labels_a: ' + str(labels_a))
            print('labels_b: ' + str(labels_b))
            images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
            pos_a, pos_b = pos_a.cuda().detach(), pos_b.cuda().detach()
            labels_a, labels_b = labels_a.cuda().detach(), labels_b.cuda().detach()

            with Timer("Elapsed time in update: %f"):
                # Main training code
                if labels_a[0] < config['ID_class_a'] and labels_b[0] < config['ID_class_a'] and config['aa']:  # aa
                    print('aa')
                    if countaa == count_dis_update:
                        trainer.dis_update_aa(images_a, images_b, config)
                        countaa = 0
                    trainer.gen_update_aa(images_a, labels_a, pos_a, images_b, labels_b, pos_b, config, subiterations)
                    countaa += 1
                elif labels_a[0] < config['ID_class_a'] and labels_b[0] >= config['ID_class_a'] and config['ab']:  # ab
                    print('ab')
                    if countab == count_dis_update:
                        trainer.dis_update_ab(images_a, images_b, config)
                        countab = 0
                    trainer.gen_update_ab(images_a, labels_a, pos_a, images_b, labels_b, pos_b, config, subiterations)
                    countab += 1
                elif labels_a[0] >= config['ID_class_a'] and labels_b[0] < config['ID_class_a'] and config['ab']:  # ba
                    print('ba')
                    if countba == count_dis_update:
                        trainer.dis_update_ab(images_b, images_a, config)
                        countba = 0
                    trainer.gen_update_ab(images_b, labels_b, pos_b, images_a, labels_a, pos_a, config, subiterations)
                    countba += 1
                elif labels_a[0] >= config['ID_class_a'] and labels_b[0] >= config['ID_class_a'] and config['bb']: # bb
                    print('bb')
                    if countbb == count_dis_update:
                        trainer.dis_update_bb(images_a, images_b, config)
                        countbb = 0
                    trainer.gen_update_bb(images_a, labels_a, pos_a, images_b, labels_b, pos_b, config, subiterations)
                    countbb += 1

                torch.cuda.synchronize()
            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                print("\033[1m Round: %02d  Epoch: %02d Iteration: %08d/%08d \033[0m \n" % (round_idx, nepoch, subiterations + 1, config['epoch_iteration'] * epoch_round), end=" ")
                write_loss(iterations, trainer, train_writer)

            iterations += 1
            subiterations += 1
            if iterations >= max_iter:
                # Save network weights
                trainer.save(checkpoint_directory, iterations)
                print('Max mAP: ' + str(max(mAP_list)*100) + '%')
                print('Max rank 1 accuracy: ' + str(max(rank1_list)*100) + '%')
                print('Max rank 5 accuracy: ' + str(max(rank5_list)*100) + '%')
                print('Max rank 10 accuracy: ' + str(max(rank10_list)*100) + '%')
                sys.exit('Finish training')
        nepoch += 1

        # test in target domain in every epoch
        trainer.test(config)
        write_loss(iterations, trainer, train_writer)
        rank1 = trainer.rank_1
        rank5 = trainer.rank_5
        rank10 = trainer.rank_10
        mAP0 = trainer.mAP_zero
        mAP05 = trainer.mAP_half
        mAPn1 = trainer.mAP_neg_one

        mAP_list.append(mAP05)
        rank1_list.append(rank1.numpy())
        rank5_list.append(rank5.numpy())
        rank10_list.append(rank10.numpy())
        # save generated images in every round
        with torch.no_grad():
            image_outputs = trainer.sample_ab(train_display_images_aba, train_display_images_abb)
        write_2images(image_outputs, display_size, image_directory, 'train_ab_%08d' % (iterations + 1))
        del image_outputs

        with torch.no_grad():
            image_outputs = trainer.sample_aa(train_display_images_aaa, train_display_images_aab)
        write_2images(image_outputs, display_size, image_directory, 'train_aa_%08d' % (iterations + 1))
        del image_outputs

        with torch.no_grad():
            image_outputs = trainer.sample_bb(train_display_images_bba, train_display_images_bbb)
        write_2images(image_outputs, display_size, image_directory, 'train_bb_%08d' % (iterations + 1))
        del image_outputs

        # regenerate data loaders in every epoch
        train_loader_a, train_loader_b, _, _ = get_mix_data_loaders(config)

        # adjust the total epochs per round
        epoch_ridx += 1
        if epoch_ridx == epoch_round and round_idx == 0:
            epoch_round = config['epoch_round']
            break

    # Save network weights
    trainer.save(checkpoint_directory, iterations)

    # update model_prev_folder
    checkpoint_directory_prev = checkpoint_directory

print('Max mAP: {:.2%}'.format(max(mAP_list)))
print('Max rank 1 accuracy: {:.2%}'.format(max(rank1_list)))
print('Max rank 5 accuracy: {:.2%}'.format(max(rank5_list)))
print('Max rank 10 accuracy: {:.2%}'.format(max(rank10_list)))
print('Finish training')
