import argparse
import logging

import torch

from src.utils import *
from torch.utils.data import DataLoader
from model.LSTM import LSTM
import torch.nn as nn
from torch import optim

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s | %(message)s',datefmt='%Y/%m/%d %H:%M:%S',level=logging.INFO)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--info',       type=str,       default="train_0811_5000_vs",  help="Information of train task")
    parser.add_argument('--model',      type=str,       default="LSTM",             help='The model of this task, eg:LSTM')
    parser.add_argument('--dataset',    type=str,       default="simulation",       help='Dataset type, default: simulation, more: real')
    parser.add_argument('--label_type', type=str,       default="vs",               help="vp or vs as label")
    parser.add_argument('--seed',       type=int,       default=2022,               help='Random seed')
    parser.add_argument('--if_noise',   type=bool,      default=True,               help='Train with noise or not')
    parser.add_argument('--data_path',  type=str,       default="data",             help='Path for storing the dataset')
    parser.add_argument('--batch_size', type=int,       default=32,                  help="Batch size of the model")
    parser.add_argument('--epochs',     type=int,       default=100,                help="Epoch numbers")
    parser.add_argument('--lr',         type=float,     default=0.0001,              help="The learning rate")
    parser.add_argument('--logspace',   type=int,       default=1,                  help="Down rate of learning rate")
    parser.add_argument('--weight_decay', type=float,   default=0.,                 help="Weight decay")
    parser.add_argument('--GPU_num',    type=str,       default="0",                help="The GPU for training")
    return parser.parse_args()




if __name__ == "__main__":

    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    PATH = f'./trained_model\\{args.info}\\{args.model}_lr_{args.lr}_epoch_{args.epochs}_bs_{args.batch_size}\\pre'
    mkdir(PATH)
    args.output_dir = PATH

    # Get dataset
    test_data = get_data(args, 'test')
    # Get dataloader
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    # Define model
    if args.model == "LSTM":
        model = LSTM(hidden_dim=128, n_layers=1, out_dim=51).to(args.device)

    logging.info("Loading model {}".format(args.model))
    args.trained_model = f'./trained_model\\{args.info}\\{args.model}_lr_{args.lr}_epoch_{args.epochs}_bs_{args.batch_size}\\trained.pth'
    model.load_state_dict(torch.load(args.trained_model, map_location=args.device))
    logging.info("Model loaded!")
    model.eval()
    pre_list, labels_list = [], []
    for i, (input_x, labels) in enumerate(test_loader):
        rdispph = input_x[0].to(device=device, non_blocking=True)
        prf = input_x[1].to(device=device, non_blocking=True)
        rwe = input_x[2].to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        rdispph = torch.ones(1, 50).to(device=device, non_blocking=True)
        rwe = torch.ones(1, 67).to(device=device, non_blocking=True)
        prf = torch.ones(1, 201).to(device=device, non_blocking=True)
        # compute output
        output = model(rdispph, prf, rwe)



        pre_list.append(output.squeeze().detach().numpy())
        labels_list.append(labels.squeeze().detach().numpy())
        #plot_wave_1(output.squeeze().detach().numpy(), name="pre")
        plot_wave_2(y1=output.squeeze().detach().numpy(), name1="pre", y2=labels.squeeze().detach().numpy(), name2="label")

    np.save(args.output_dir+"/pre_list.npy", pre_list)
    np.save(args.output_dir+"/labels_list.npy", labels_list)
    print("over")




