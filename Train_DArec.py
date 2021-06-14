import random
import os
import numpy as np
import torch.optim as optim
import torch.utils.data
from function import *
from model import DArec
from torch.utils.data import DataLoader
from Data_Preprocessing import Mydata
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
parser = argparse.ArgumentParser(description='DArec with PyTorch')
parser.add_argument('--epochs', '-e', type=int, default=20)
parser.add_argument('--batch_size', '-b', type=int, default=32)
parser.add_argument('--lr', '-l', type=float, help='learning rate', default=1e-3)
parser.add_argument('--wd', '-w', type=float, help='weight decay(lambda)', default=2e-5)
parser.add_argument("--n_factors", type=int, default=400, help="embedding dim")
parser.add_argument("--n_users", type=int, default=1637, help="size of each image batch")
parser.add_argument("--S_n_items", type=int, default=23450, help="Source items number")
parser.add_argument("--T_n_items", type=int, default=16993, help="Target items number")
parser.add_argument("--RPE_hidden_size", type=int, default=500, help="hidden size of Rating Pattern Extractor")
parser.add_argument("--S_pretrained_weights", type=str, default=r'D:\DARec\Pretrained_ParametersS_AutoRec_100.pkl')
parser.add_argument("--T_pretrained_weights", type=str, default=r'D:\DARec\Pretrained_ParametersT_AutoRec_100.pkl')
args = parser.parse_args()

### Load Data
train_dataset = Mydata(r'D:\DARec\Dataset\ratings_Toys_and_Games.csv', r'D:\DARec\Dataset\ratings_Automotive.csv', train=True)
test_dataset = Mydata(r'D:\DARec\Dataset\ratings_Toys_and_Games.csv', r'D:\DARec\Dataset\ratings_Automotive.csv', train=False)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

args.n_users = train_dataset.S_data.shape[0]
args.S_n_items, args.T_n_items = train_dataset.S_data.shape[1], train_dataset.T_data.shape[1]

print("Data is loaded")
### neural network

net = DArec(args)
net.S_autorec.load_state_dict(torch.load(args.S_pretrained_weights))
net.T_autorec.load_state_dict(torch.load(args.T_pretrained_weights))
net.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), weight_decay=args.wd, lr=args.lr)
RMSE = MRMSELoss().cuda()
criterion = DArec_Loss().cuda()
# loss, source_mask, target_mask
def train(epoch):
    process = []
    Total_Loss = 0
    for idx, d in enumerate(train_loader):
        ### alpha参考DDAN
        p = float(idx + epoch * len(train_loader)) / args.batch_size / len(train_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        ### 数据
        source_rating, target_rating, source_labels, target_labels = d

        source_rating = source_rating.cuda()
        target_rating = target_rating.cuda()
        source_labels = source_labels.squeeze(1).long().cuda()
        target_labels= target_labels.squeeze(1).long().cuda()
        # print(next(net.S_autorec.parameters()).device)
        # print(source_rating.device)
        optimizer.zero_grad()
        is_source = True
        if is_source == True:
            class_output, source_prediction, target_prediction = net(source_rating, alpha, is_source)
            source_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                    source_rating, target_rating, source_labels)

            rmse, _ = RMSE(target_prediction, target_rating)
            rmse = torch.sqrt(rmse.item() / torch.sum(target_mask))

            loss = source_loss
        is_source = False
        if is_source == False:
            class_output, source_prediction, target_prediction = net(target_rating, alpha, is_source)
            target_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                              source_rating, target_rating, target_labels)
            loss += target_loss

        loss.backward()
        optimizer.step()

        process.append(rmse)
    return torch.tensor(sum(process) / (len(process))).detach().cpu()

def test():
    process = []
    with torch.no_grad():
        for idx, d in enumerate(test_loader):
            ### alpha参考DDAN
            p = float(idx + epoch * len(train_loader)) / args.batch_size / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            ### 数据
            source_rating, target_rating, source_labels, target_labels = d

            source_rating = source_rating.cuda()
            target_rating = target_rating.cuda()
            source_labels = source_labels.squeeze(1).long().cuda()
            target_labels = target_labels.squeeze(1).long().cuda()

            # optimizer.zero_grad()
            is_source = True
            if is_source == True:
                class_output, source_prediction, target_prediction = net(source_rating, alpha, is_source)
                source_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                        source_rating, target_rating, source_labels)
                rmse, _ = RMSE(target_prediction, target_rating)
                rmse = torch.sqrt(rmse.item() / torch.sum(target_mask))

                loss = source_loss
            # is_source = False
            # if is_source == False:
            #     class_output, source_prediction, target_prediction = net(target_rating, alpha, is_source)
            #     target_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
            #                                                       source_rating, target_rating, source_class)
            #     loss += target_loss


        process.append(rmse)
    return torch.tensor(sum(process) / (len(process))).detach().cpu()

if __name__=="__main__":
    train_rmse = []
    test_rmse = []
    wdir = r"D:\DARec\DARec_"

    for epoch in tqdm(range(args.epochs)):
        train_rmse.append(train(epoch))
        test_rmse.append(test())
        if epoch % 20 == 19:
            torch.save(net.state_dict(), wdir+"%d.pkl" % (epoch+1))
    plt.plot(range(args.epochs), train_rmse, range(args.epochs), test_rmse)
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.xticks(range(0, args.epochs, 2))
    plt.show()
