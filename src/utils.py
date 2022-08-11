import random

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import MultiSignalDatasets


def get_data(args, split='train'):
    data_path = os.path.join(args.data_path, args.dataset) + f'_{split}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = MultiSignalDatasets(args.data_path, args.dataset, split, args.if_noise, label=args.label_type)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'pre_trained_models/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model


def set_random(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_wave_2(y1,name1,y2,name2):

    fig, ax = plt.subplots(figsize=(4, 3))
    line1, = ax.plot(y1, label='{}'.format(name1), )
    line2, = ax.plot(y2, "r--", label='{}'.format(name2))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.legend(fontsize=16)
    plt.show()


def plot_wave_3(y1,name1,y2,name2,y3,name3):

    fig, ax = plt.subplots(figsize=(25, 8))
    line1, = ax.plot(y1, label='{}'.format(name1))
    line2, = ax.plot(y2, label='{}'.format(name2))
    line3, = ax.plot(y3, label='{}'.format(name3))
    ax.legend()
    plt.show()


def plot_wave_1(y, name):
    fontsize = 16
    fig, ax = plt.subplots(figsize=(4, 3))
    line1, = ax.plot(y, label='{}'.format(name))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.legend(fontsize=fontsize)
    plt.show()


def draw_Figure_acc(train_loss_list, val_loss_list, mode, args):
    x = np.arange(0, len(train_loss_list), 1)
    y1 = np.array(train_loss_list)
    y0 = np.array(val_loss_list)

    plt.figure(figsize=(8, 7))
    plt.plot(x, y0, color="blue", label="Val")
    plt.plot(x, y1, color="red", label="Train")

    plt.xlabel("epoch")
    plt.ylabel("{}".format(mode))
    plt.legend(loc='best', ncol=1)
    plt.savefig(args.output_dir + "/{}_curve.jpg".format(mode)), plt.close()


def save_hparam(args, path):
    savepath = path+ "/hparam.txt"
    with open(savepath, "w") as fp:
        args_dict = args.__dict__
        for key in args_dict:
            fp.write("{} : {}\n".format(key, args_dict[key]))


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


