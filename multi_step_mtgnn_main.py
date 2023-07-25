import os
import sys
import argparse
import json

import torch
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.utils import *
from utils.torchUtils import *
from layers.models import *
from utils.dataloader import *

args = argparse.Namespace(
    # data path
    data_type='skt',
    data_path='./data/skt',
    tr=0.7,
    val=0.2,
    standardize=True,
    exclude_TA=True,
    lag=7,
    cache_file='./data/cache.pickle',
    # training options
    batch_size=32,
    epoch=30,
    lr=0.001,
    kl_loss_penalty=0.01,
    patience=30,
    delta=0.,
    print_log_option=10,
    verbose=True,
    reg_loss_penalty=1e-2,
    kl_weight=0.1,
    gradient_max_norm=5,
    # model options
    model_path='./data/skt/multi_mtgnn',
    num_blocks=3,
    k=2,
    top_k=4,
    embedding_dim=128,
    alpha=3,
    beta=0.5,
    tau=1.,
    hard=True,
    # To test
    test=False,   #학습하고 싶을땐 False
    model_file='latest_checkpoint.pth.tar',
    model_type='proto',
    num_folds=1,
    pred_steps=3

)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make a path to save a model
if not os.path.exists(args.model_path):
    print("Making a path to save the model...")
    os.makedirs(args.model_path, exist_ok=True)
else:
    print("The path already exists, skip making the path...")

print(f'saving the commandline arguments in the path: {args.model_path}...')
args_file = os.path.join(args.model_path, 'commandline_args.txt')
with open(args_file, 'w') as f:
    json.dump(args.__dict__, f, indent=2)


def main(args):
    # read data
    print("Loading data...")
    if args.data_type == 'skt':
        # load gestures-data
        # data는 data['train'] 이 X와 M으로 나뉘는데 X가 (enb개수, 관측시점개수,col개수)임.
        data = load_skt(args) if not args.exclude_TA else load_skt_without_TA(args)
    else:
        print("Unkown data type, data type should be \"skt\"")
        sys.exit()

    # define training, validation, test datasets and their dataloaders respectively
    train_data, valid_data, test_data \
        = TimeSeriesDataset(*data['train'], lag= args.lag,pred_steps=args.pred_steps),\
          TimeSeriesDataset(*data['valid'], lag= args.lag,pred_steps=args.pred_steps),\
          TimeSeriesDataset(*data['test'], lag= args.lag,pred_steps=args.pred_steps)
    train_loader, valid_loader, test_loader \
        = DataLoader(train_data, batch_size = args.batch_size, shuffle = False),\
            DataLoader(valid_data, batch_size = args.batch_size, shuffle = False),\
            DataLoader(test_data, batch_size = args.batch_size, shuffle = False)

    print("Loading data done!")

    model = Multistep_MTGNN(
        num_heteros=args.num_heteros,
        num_ts=args.num_ts,
        time_lags=args.lag,
        num_blocks=args.num_blocks,
        k=args.k,
        embedding_dim=args.embedding_dim,
        device=device,
        alpha=args.alpha,
        top_k=args.top_k,
        pred_steps=args.pred_steps
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), args.lr)
    early_stopping = EarlyStopping(
        patience=args.patience,
        delta=args.delta,
        path=args.model_path)

    if args.test:
        model_file = os.path.join(args.model_path, args.model_file)  # .data/skt/model/latest_checkpoint.path.tar 이 되는거임.
        ckpt = torch.load(model_file)
        model.load_state_dict(ckpt['state_dict'])
    else:
        multistep_mtgnn_train(args, model, train_loader, valid_loader, optimizer, criterion, early_stopping, device)


    print(f"Testing the model...")
    perf = multistep_mtgnn_test_regr(args, model, test_loader, criterion, device)
    return perf


#main(args)
# train_loader,model = main(args)
#
# a=iter(train_loader)
# b=a.next()
# out=model.forward(b,beta=0.5)
# print(out['outs_label'].shape)
# print(b['label'].shape)
# tmp_out= nn.Sequential(nn.Conv2d(306, 11, kernel_size=(7-args.pred_steps+1, 1), padding=0),nn.Tanh())
# result =  tmp_out(after_fc_decode )
# print(result.shape)
perf = main(args)
