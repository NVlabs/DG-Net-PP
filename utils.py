"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from data import ImageFilelist
from torchvision.datasets import ImageFolder
from reIDfolder import ReIDFolder, ReIDFolder_mix
import torch
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
from random import shuffle
import torch.nn.init as init
import time
from operator import itemgetter
from shutil import rmtree
# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_data_loader_folder    : folder-based data loader
# get_config                : load yaml file
# eformat                   :
# write_2images             : save output image
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# prepare_sub_folder_pseudo : create checkpoints, images and pseudo labels folders for saving outputs
# get_mix_data_loaders
# write_one_row_html        : write one row of the html file for output images
# write_html                : create the html file.
# write_loss
# slerp
# get_slerp_interp
# get_model_list
# load_vgg16
# vgg_preprocess
# get_scheduler
# weights_init

def get_mix_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    if 'new_size' in conf:
        new_size_a= conf['new_size']
        new_size_b = conf['new_size']
    else:
        new_size_a = conf['new_size_a']
        new_size_b = conf['new_size_b']
    height = conf['crop_image_height']
    width = conf['crop_image_width']

    # generate the list of the mixed data folder
    train_path = conf['data_root']
    mixData = ImageFolder(train_path)
    ab_list = mixData.imgs
    ab_idx = [i for i in range(len(ab_list))]
    size_a = conf['sample_a']
    size_b = conf['sample_b']
    # full lists of two datasets
    a_full_idx = ab_idx[0:size_a]
    b_full_idx = ab_idx[size_a:]
    # generate two sample lists of two datasets with equal size
    if size_a > size_b:
        sel_idx = list(np.random.choice(size_a, size_b, replace=False))
        a_idx = list(itemgetter(*sel_idx)(a_full_idx))
        b_idx = b_full_idx.copy()
    elif size_b > size_a:
        sel_idx = list(np.random.choice(size_b, size_a, replace=False))
        b_idx = list(itemgetter(*sel_idx)(b_full_idx))
        a_idx = a_full_idx.copy()
    else:
        a_idx = a_full_idx.copy()
        b_idx = b_full_idx.copy()

    a_idx_a = a_idx.copy()
    a_idx_b = a_idx.copy()
    b_idx_a = b_idx.copy()
    b_idx_b = b_idx.copy()

    # generate two lists for train_loader_a and train_loader_b
    ab_port = conf['ab_port']
    bs = conf['batch_size']
    size_domain = min(size_a, size_b)
    ab_num = math.floor(ab_port * size_domain) // bs * bs
    xx_num = (size_domain - ab_num) // bs * bs
    idx_la = [] # list for loader_a
    idx_lb = [] # list for loader_b

    sel_idx_ab_a = list(np.random.choice(size_domain, ab_num, replace=False))
    sel_idx_ab_b = list(np.random.choice(size_domain, ab_num, replace=False))
    sel_idx_ba_a = list(np.random.choice(size_domain, ab_num, replace=False))
    sel_idx_ba_b = list(np.random.choice(size_domain, ab_num, replace=False))

    aa_idx_a = [a_idx_a[i] for i in range(size_domain) if i not in sel_idx_ab_a]  # batch aa for train_loader_a
    aa_idx_b = [a_idx_b[i] for i in range(size_domain) if i not in sel_idx_ba_b]  # batch aa for train_loader_b
    bb_idx_a = [b_idx_a[i] for i in range(size_domain) if i not in sel_idx_ba_a]  # batch bb for train_loader_a
    bb_idx_b = [b_idx_b[i] for i in range(size_domain) if i not in sel_idx_ab_b]  # batch bb for train_loader_b
    shuffle(aa_idx_a)
    shuffle(aa_idx_b)
    shuffle(bb_idx_a)
    shuffle(bb_idx_b)
    aa_idx_a = aa_idx_a[:xx_num]
    aa_idx_b = aa_idx_b[:xx_num]
    bb_idx_a = bb_idx_a[:xx_num]
    bb_idx_b = bb_idx_b[:xx_num]
    ab_idx_a, ab_idx_b, ba_idx_a, ba_idx_b = [], [], [], []
    if sel_idx_ab_a != []:
        ab_idx_a = list(itemgetter(*sel_idx_ab_a)(a_idx_a))  # batch ab for train_loader_a
    if sel_idx_ab_b != []:
        ab_idx_b = list(itemgetter(*sel_idx_ab_b)(b_idx_b))  # batch ab for train_loader_b
    if sel_idx_ba_a != []:
        ba_idx_a = list(itemgetter(*sel_idx_ba_a)(b_idx_a))  # batch ab for train_loader_a
    if sel_idx_ba_b != []:
        ba_idx_b = list(itemgetter(*sel_idx_ba_b)(a_idx_b))  # batch ab for train_loader_b

    aa_thresh = conf['xx_port'] / 2
    bb_thresh = aa_thresh * 2
    ab_thresh = bb_thresh + conf['ab_port'] / 2
    while aa_idx_b or bb_idx_a or ab_idx_a or ba_idx_a:
        dice = np.random.uniform(0, 1)
        if dice <= aa_thresh:
            if not aa_idx_a:
                continue
            for _ in range(batch_size):
                idx_la.append(aa_idx_a.pop())
                idx_lb.append(aa_idx_b.pop())
        elif dice > aa_thresh and dice <= bb_thresh:
            if not bb_idx_a:
                continue
            for _ in range(batch_size):
                idx_la.append(bb_idx_a.pop())
                idx_lb.append(bb_idx_b.pop())
        elif dice > bb_thresh and dice <= ab_thresh:
            if not ab_idx_a:
                continue
            for _ in range(batch_size):
                idx_la.append(ab_idx_a.pop())
                idx_lb.append(ab_idx_b.pop())
        else:
            if not ba_idx_a:
                continue
            for _ in range(batch_size):
                idx_la.append(ba_idx_a.pop())
                idx_lb.append(ba_idx_b.pop())

    train_loader_a = get_data_loader_folder_mix(os.path.join(conf['data_root'], 'train_all'), idx_la, batch_size, True,
                                          new_size_a, height, width, num_workers, True)
    test_loader_a = get_data_loader_folder(os.path.join(conf['data_root_a'], 'query'), batch_size, False,
                                         new_size_a, height, width, num_workers, False)
    train_loader_b = get_data_loader_folder_mix(os.path.join(conf['data_root'], 'train_all'), idx_lb, batch_size, True,
                                          new_size_b, height, width, num_workers, True)
    test_loader_b = get_data_loader_folder(os.path.join(conf['data_root_b'], 'query'), batch_size, False,
                                         new_size_b, height, width, num_workers, False)

    return train_loader_a, train_loader_b, test_loader_a, test_loader_b

def get_all_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    if 'new_size' in conf:
        new_size_a= conf['new_size']
        new_size_b = conf['new_size']
    else:
        new_size_a = conf['new_size_a']
        new_size_b = conf['new_size_b']
    height = conf['crop_image_height']
    width = conf['crop_image_width']

    if 'data_root' in conf:
        train_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'train_all'), batch_size, True,
                                              new_size_a, height, width, num_workers, True)
        test_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'query'), batch_size, False,
                                             new_size_a, height, width, num_workers, False)
        train_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'train_all'), batch_size, True,
                                              new_size_b, height, width, num_workers, True)
        test_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'query'), batch_size, False,
                                             new_size_b, height, width, num_workers, False)
    else:
        train_loader_a = get_data_loader_list(conf['data_folder_train_a'], conf['data_list_train_a'], batch_size, True,
                                                new_size_a, height, width, num_workers, True)
        test_loader_a = get_data_loader_list(conf['data_folder_test_a'], conf['data_list_test_a'], batch_size, False,
                                                new_size_a, height, width, num_workers, False)
        train_loader_b = get_data_loader_list(conf['data_folder_train_b'], conf['data_list_train_b'], batch_size, True,
                                                new_size_b, height, width, num_workers, True)
        test_loader_b = get_data_loader_list(conf['data_folder_test_b'], conf['data_list_test_b'], batch_size, False,
                                                new_size_b, height, width, num_workers, False)
    return train_loader_a, train_loader_b, test_loader_a, test_loader_b


def get_data_loader_list(root, file_list, batch_size, train, new_size=None,
                           height=256, width=128, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.485, 0.456, 0.406),  
                                           (0.229, 0.224, 0.225))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Pad(10, padding_mode='edge')] + transform_list if train else transform_list
    transform_list = [transforms.Resize((height, width), interpolation=3)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFilelist(root, file_list, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    return loader

def get_data_loader_folder(input_folder, batch_size, train, new_size=None,
                           height=256, width=128, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.485, 0.456, 0.406),
                                           (0.229, 0.224, 0.225))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Pad(10, padding_mode='edge')] + transform_list if train else transform_list
    transform_list = [transforms.Resize((height,width), interpolation=3)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ReIDFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    return loader

def get_data_loader_folder_mix(input_folder, idx_list, batch_size, train, new_size=None,
                           height=256, width=128, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.485, 0.456, 0.406),
                                           (0.229, 0.224, 0.225))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Pad(10, padding_mode='edge')] + transform_list if train else transform_list
    transform_list = [transforms.Resize((height,width), interpolation=3)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ReIDFolder_mix(input_folder, transform=transform, idx_list=idx_list)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
    return loader

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def eformat(f, prec):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d"%(mantissa, int(exp))


def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True, scale_each=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    __write_images(image_outputs[0:n//2], display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[n//2:n], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))

def vis_2images(image_outputs, display_image_num, image_directory, postfix):
    __write_images(image_outputs, display_image_num, '%s/batch_%s.jpg' % (image_directory, postfix))

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory

def prepare_sub_folder_pseudo(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    pseudo_directory = os.path.join(output_directory, 'pseudo_train')
    if not os.path.exists(pseudo_directory):
        print("Creating directory: {}".format(pseudo_directory))
        os.makedirs(pseudo_directory)
        os.makedirs(pseudo_directory + '/train_all')
    else:
        rmtree(pseudo_directory)
        os.makedirs(pseudo_directory)
        os.makedirs(pseudo_directory + '/train_all')

    return checkpoint_directory, image_directory, pseudo_directory


def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations,img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    write_one_row_html(html_file, iterations, '%s/gen_b2a_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations-1, -1):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/gen_a2b_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'rank_' in attr or 'mAP_' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
            os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
        vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean)) # subtract mean
    return batch


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    elif hyperparameters['lr_policy'] == 'multistep':
        #50000 -- 75000 -- 
        step = hyperparameters['step_size']
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[step, step+step//2, step+step//2+step//4],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


