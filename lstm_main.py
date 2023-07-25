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
    model_path='.data/skt/model',
    num_blocks=3,
    k=2,
    top_k=4,
    embedding_dim=128,
    alpha=3,
    beta=0.5,
    tau=1.,
    hard=True,
    # To test
    test=False,   #학습하고 싶을땐 True로 바꿔
    model_file='latest_checkpoint.pth.tar',
    model_type='proto',
    num_folds=1

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
    # Mask는 LSTM에서 사용하지 않음

    perfs= dict()

    for enb_idx in range(data['train'][0].shape[0]):
        args.enb = args.decoder.get(enb_idx)
        perfs[args.enb] = dict()

        enb_data_train = data['train'][0][enb_idx]
        enb_data_valid = data['valid'][0][enb_idx]
        enb_data_test = data['test'][0][enb_idx]

        for col_idx, col in enumerate(args.columns):
            args.col = col #test_regr에서 파일명에 넣을 용도
            train_sequences = create_sequences(pd.DataFrame(enb_data_train,columns=args.columns), target_column=col, sequence_length=args.lag)
            valid_sequences = create_sequences(pd.DataFrame(enb_data_valid,columns=args.columns), target_column=col, sequence_length=args.lag)
            test_sequences = create_sequences(pd.DataFrame(enb_data_test,columns=args.columns), target_column=col, sequence_length=args.lag)

            # getitem으로 얻는 shape는 train_data['input']은 (lag값,col개수)
            # getitem으로 얻는 shape는 train['label']은 (pred_steps=1,target col1개)
            train_data = LSTMDataset(train_sequences)
            valid_data = LSTMDataset(valid_sequences)
            test_data = LSTMDataset(test_sequences)

            #loader에서 한 iter는 ['sequence']는 (batchsize,lag값,col개수) shape가짐.
            #loader에서 한 iter는 ['label']는 (batchsize,pred_steps=1,target col1개) shape가짐.
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
            valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)



            model = MyLSTM(n_features=len(args.columns), n_hidden=128, n_layers=2, drop_out=0.2).to(device)




            if enb_idx == 0 & col_idx == 0:
                print('The model is on GPU') if next(model.parameters()).is_cuda else print('The model is on CPU')

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), args.lr)
            early_stopping = EarlyStopping(
                patience=args.patience,
                delta=args.delta,
                path=args.model_path)                        #.data/skt/model/

            if args.test:
                model_file = os.path.join(args.model_path, args.model_file)  #.data/skt/model/latest_checkpoint.path.tar 이 되는거임.
                ckpt = torch.load(model_file)
                model.load_state_dict(ckpt['state_dict'])
            else:
                lstm_train(args, model, train_loader, valid_loader, optimizer, criterion, early_stopping, device)


            print(f"enb_{args.enb}_{args.col}_Testing the model...")
            perf = lstm_test_regr(args, model, test_loader, criterion, device)
            perfs[args.enb][args.col] = perf

    return perfs













# train_loader, valid_loader, test_loader, model = main(args)
#
# for batch_idx, x in enumerate(train_loader):
#     x['sequence'], x['label'] = x['sequence'].to(device), x['label'].to(device)
# model.train()  # 이제 학습할거라고 알려주는 부분
# # with torch.set_grad_enabled(True):
# out = model.forward(x)



#main에서 if args.test: 부분
#lstm_test_regr에서도 모든 loader반복 막기 위해 break문 넣은거 생각!
perf = main(args)
#test_loader = main(args)

