import argparse
import logging

from src.utils import *
from torch.utils.data import DataLoader
from model.LSTM import LSTM
import torch.nn as nn
from torch import optim

parser = argparse.ArgumentParser()
parser.add_argument('--info',           type=str,   default="train_0810_5000_vs",  help="Information of train task")
parser.add_argument('--model',          type=str,   default="LSTM",         help='The model of this task, eg:LSTM')
parser.add_argument('--dataset',        type=str,   default="simulation",   help='Dataset type, default: simulation, more: real')
parser.add_argument('--label_type',     type=str,   default="vs",           help="vp or vs as label")
parser.add_argument('--seed',           type=int,   default=2022,           help='Random seed')
parser.add_argument('--if_noise',       type=bool,  default=True,           help='Train with noise or not')
parser.add_argument('--data_path',      type=str,   default="data",         help='Path for storing the dataset')
parser.add_argument('--batch_size',     type=int,   default=64,              help="Batch size of the model")
parser.add_argument('--epochs',         type=int,   default=100,            help="Epoch numbers")
parser.add_argument('--lr',             type=float, default=0.001,          help="The learning rate")
parser.add_argument('--logspace',       type=int,   default=1,              help="Down rate of learning rate")
parser.add_argument('--weight_decay',   type=float, default=0.,             help="Weight decay")
parser.add_argument('--GPU_num',        type=str,   default="0",            help="The GPU for training")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s | %(message)s',datefmt='%Y/%m/%d %H:%M:%S',level=logging.INFO)


def network(args):
    if args.seed is not None:
        set_random(args.seed+args.epochs+args.batch_size)

    train_data = get_data(args, 'train')
    test_data = get_data(args, 'test')
    val_data = get_data(args, 'val')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

    # logging info
    args_dict = args.__dict__
    for key in args_dict:
        logging.info("{} : {}\n".format(key, args_dict[key]))
    save_hparam(args, path=args.output_dir)

    # Define model
    if args.model == "LSTM":
        model = LSTM(embed_dim=128, hidden_dim=128, n_layers=1, out_dim=51).to(args.device)

    # Define loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    torch.cuda.empty_cache()
    train_loss_list, val_loss_list = [], []

    for epoch in range(0, args.epochs):
        #学习率梯度下降
        lr=[]
        if args.logspace != 0:
            logspace_lr = np.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = logspace_lr[epoch]
                lr += [param_group['lr']]
        print("[epoch] {}  [lr]: {}".format(epoch + 1, lr))


        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        val_loss = validate(val_loader, model, criterion, epoch)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        draw_Figure_acc(train_loss_list, val_loss_list, mode="Loss", args=args)

        torch.cuda.empty_cache()
        if epoch%20 == 0:
            torch.save(model.state_dict(), args.output_dir + '/trained_{}.pth'.format(str(epoch)))

    torch.save(model.state_dict(),args.output_dir + '/trained.pth')
    np.save(args.output_dir+"/train_loss.npy", train_loss_list)
    np.save(args.output_dir+"/val_loss.npy", val_loss_list)

    logging.info("Model Saved!")


def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    Loss = 0

    for i, (input_x, labels) in enumerate(train_loader):
        rdispph = input_x[0].to(device=device, non_blocking=True)
        prf = input_x[1].to(device=device, non_blocking=True)
        rwe = input_x[2].to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)


        # compute output
        output = model(rdispph, prf, rwe)
        plot_wave_2(y1=output.squeeze().detach().numpy()[1], name1="pre", y2=labels.squeeze().detach().numpy()[1], name2="label")
        loss = criterion(output.squeeze(), labels.squeeze())


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Loss += loss.cpu().item()

        if i % 10 == 0:
            print("[epoch] %d [train] batch: %d, loss: %.5f" % (epoch + 1, i, Loss / (i + 1)))
    Loss= Loss / (i + 1)
    return Loss


def validate(val_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    Loss = 0

    for i, (input_x, labels) in enumerate(val_loader):
        rdispph = input_x[0].to(device=device, non_blocking=True)
        prf = input_x[1].to(device=device, non_blocking=True)
        rwe = input_x[2].to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        # compute output
        output = model(rdispph, prf, rwe)
        loss = criterion(output.squeeze(), labels.squeeze())

        Loss += loss.cpu().item()


        if i % 10 == 0:
            print("[epoch] %d [val] batch: %d, loss: %.5f" % (epoch + 1, i, Loss / (i + 1)))
    Loss = Loss / (i + 1)
    return Loss


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_num

    PATH = './trained_model/{0}/{4}_lr_{1}_epoch_{2}_bs_{3}'.format(args.info, args.lr, args.epochs, args.batch_size, args.model)
    mkdir(PATH)
    args.output_dir = PATH
    network(args)