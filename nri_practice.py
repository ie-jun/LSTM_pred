import os
import sys
import argparse
import json

import pandas as pd
from time import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.utils import *
from utils.torchUtils import *
from layers.models import *
from layers.graphLearningLayers import *
from layers.nriLayers import *
from utils.dataloader import *

#parser = argparse.ArgumentParser()

# # Data path
# parser.add_argument('--data_type', type=str, default='skt',
#                     help='one of: skt')
# parser.add_argument('--data_path', type=str, default='./data/skt')
# parser.add_argument('--pred_steps', type=int, default=3,
#                     help='the number of steps to predict')
# parser.add_argument('--tr', type=float, default=0.7,
#                     help='the ratio of training data to the original data')
# parser.add_argument('--val', type=float, default=0.2,
#                     help='the ratio of validation data to the original data')
# parser.add_argument('--standardize', action='store_true',
#                     help='standardize the inputs if it is true.')
# parser.add_argument('--exclude_TA', action='store_true',
#                     help='exclude TA column if it is set true.')
# parser.add_argument('--lag', type=int, default=1,
#                     help='time-lag (default: 1)')
# parser.add_argument('--cache_file', type=str, default='./data/cache.pickle',
#                     help='a cache file to min-max scale the data')
#
# # Training options
# parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
# parser.add_argument('--fine_tunning_every', type=int, default=12,
#                     help='fine tune a model every \'fine_tunning_every\'')
# parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
# parser.add_argument('--epoch_online', type=int, default=30, help='the number of epoches to train online for')
# parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
# parser.add_argument('--kl_loss_penalty', type=float, default=0.01, help='kl-loss penalty (default= 0.01)')
# parser.add_argument('--patience', type=int, default=5, help='patience of early stopping condition')
# parser.add_argument('--delta', type=float, default=0.01, help='significant improvement to update a model')
# parser.add_argument('--print_log_option', type=int, default=10, help='print training loss every print_log_option')
# parser.add_argument('--verbose', action='store_true',
#                     help='print logs about early-stopping')
# parser.add_argument('--train_ar', action='store_true',
#                     help='train autoregressive predictions if set true')
# parser.add_argument('--train_online', action='store_true',
#                     help='train online if set true')
#
# # model options
# parser.add_argument('--model_path', type=str, default='./data/skt/model',
#                     help='a path to (save) the model')
# parser.add_argument('--num_blocks', type=int, default=3,
#                     help='the number of the HeteroBlocks (default= 3)')
# parser.add_argument('--k', type=int, default=2,
#                     help='the number of layers at every GC-Module (default= 2)')
# parser.add_argument('--top_k', type=int, default=4,
#                     help='top_k to select as non-zero in the adjacency matrix    (default= 4)')
# parser.add_argument('--embedding_dim', type=int, default=128,
#                     help='the size of embedding dimesion in the graph-learning layer (default= 128)')
# parser.add_argument('--beta', type=float, default=0.5,
#                     help='parameter used in the GraphConvolutionModule, must be in the interval [0,1] (default= 0.5)')
#parser.add_argument('--tau', type= float, default= 1.,
#                help= 'smoothing parameter used in the Gumbel-Softmax, only used in the model: heteroNRI')
# parser.add_argument('--model_name', type=str, default='latest_checkpoint.pth.tar'
#                     , help='model name to save', required=False)
# parser.add_argument('--n_hid_encoder', type=int, default=256,
#                     help='dimension of a hidden vector in the nri-encoder')
# parser.add_argument('--msg_hid', type=int, default=256,
#                     help='dimension of a message vector in the nri-decoder')
# parser.add_argument('--msg_out', type=int, default=256,
#                     help='dimension of a message vector (out) in the nri-decoder')
# parser.add_argument('--n_hid_decoder', type=int, default=256,
#                     help='dimension of a hidden vector in the nri-decoder')
#
# # to save predictions and graphs
# parser.add_argument('--save_results', action='store_true',
#                     help='to save graphs and figures in the model_path')
# args = parser.parse_args()


args = argparse.Namespace(
    # data path
    data_type='skt',
    data_path='./data/skt',
    pred_steps=3,
    tr=0.7,
    val=0.2,
    standardize=True, #action='store_true'
    exclude_TA=True,  #action='store_true'
    lag=7,
    cache_file='./data/cache.pickle',
    # training options
    batch_size=2,
    fine_tunning_every=12, #nri multistep에 처음생김
    epoch=30,
    epoch_online=30, #nri multistep에 처음생김
    lr=0.001,
    kl_loss_penalty=0.01,
    patience=5,
    delta=0.01,
    print_log_option=10,
    verbose=True,  #action='store_true'
    train_ar = True, #action='store_true'이고 아마 이게 True
    train_online = False, #action='store_true'이고 아마 이게 False
    # reg_loss_penalty=1e-2,
    # kl_weight=0.1,
    # gradient_max_norm=5,

    # model options
    model_path='.data/skt/model',
    num_blocks=3,
    k=2,
    top_k=4,
    embedding_dim=128,
    #alpha=3,
    beta=0.5,
    tau=1.,
    model_name='latest_checkpoint.pth.tar',
    n_hid_encoder=256,
    msg_hid=256,
    msg_out=256,
    n_hid_decoder=256,
    save_result = True #action='store_true'
    #hard=True,
    # To test
    #test=False,   #학습하고 싶을땐 True로 바꿔
    #model_file='latest_checkpoint.pth.tar',
    #model_type='proto',
    #num_folds=1

)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make a path to save a model
if not os.path.exists(args.model_path):
    print("Making a path to save the model...")
    os.makedirs(args.model_path, exist_ok=True)
else:
    print("The path already exists, skip making the path...")

print(f'saving the commandline arguments in the path: {args.model_path}...')
args_file = os.path.join(args.model_path, 'commandline_args(online).txt')
with open(args_file, 'w') as f:
    json.dump(args.__dict__, f, indent=2)


def main(args):
    # read data
    print("Loading data...")
    if args.data_type == 'skt':
        # load gestures-data
        data = load_skt(args) if not args.exclude_TA else load_skt_without_TA(args)
    else:
        print("Unkown data type, data type should be \"skt\"")
        sys.exit()

    # define training, validation, test datasets and their dataloaders respectively
    train_data, valid_data, test_data \
        = TimeSeriesDataset(*data['train'], lag=args.lag, pred_steps=args.pred_steps), \
          TimeSeriesDataset(*data['valid'], lag=args.lag, pred_steps=args.pred_steps), \
          TimeSeriesDataset(*data['test'], lag=args.lag, pred_steps=args.pred_steps)
    train_loader, valid_loader, test_loader \
        = DataLoader(train_data, batch_size=args.batch_size, shuffle=False), \
          DataLoader(valid_data, batch_size=args.batch_size, shuffle=False), \
          DataLoader(test_data, batch_size=args.fine_tunning_every, shuffle=False)

    print("Loading data done!")

    model = NRIMulti(
        num_heteros=args.num_heteros,
        num_time_series=args.num_ts,
        time_lags=args.lag,
        #device=device,
        tau=args.tau,
        n_hid_encoder=args.n_hid_encoder,
        msg_hid=args.msg_hid,
        msg_out=args.msg_out,
        n_hid_decoder=args.n_hid_decoder,
        pred_steps=args.pred_steps,
        device=device
    ).to(device)
    return train_loader, model
    # setting training args...
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)
    # optimizer.load_state_dict(ckpt['optimizer'])
    early_stopping = EarlyStopping(
        patience= args.patience,
        verbose= args.verbose,
        delta = args.delta,
        path= args.model_path,
        model_name= args.model_name
    )

    # train the multi-step heads using the training data
    if args.train_ar:
        train(args, model, train_loader, valid_loader, optimizer, criterion, early_stopping, device)
    else:
        print('skip training auto-regressives predictions...')
    # test the multi-step HeteroNRI (model) using test data
    # fine tune the model every 'args.fine_tunning_every'
    # test the fine-tuned model using the next batch of the test dataset.

    if args.train_online:
        print('start online-learning...')
    else:
        print('start evaluating...')
    # record time elapsed of the fine-tunning
    # the time should be less than 5 minutes...


train_loader, model = main(args)

a=iter(train_loader)
b=a.next()
c=model.forward(b,args.beta)
#
# # out = main(args)
# # a= 1+2