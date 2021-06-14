import torch
from torch import nn, optim
from AutoRec import AutoRec
from Data_Preprocessing import Mydata
from function import MRMSELoss
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
切换不同autoencoder需要修改的地方：
1. n_items这个
2. 主函数中保存路径
3. train和test中data[1]
"""

def check_positive(val):
    val = int(val)
    if val <=0:
        raise argparse.ArgumentError(f'{val} is invalid value. epochs should be positive integer')
    return val

parser = argparse.ArgumentParser(description='AutoRec with PyTorch')
parser.add_argument('--epochs', '-e', type=check_positive, default=100)
parser.add_argument('--batch_size', '-b', type=check_positive, default=32)
parser.add_argument('--lr', '-l', type=float, help='learning rate', default=1e-3)
parser.add_argument('--wd', '-w', type=float, help='weight decay(lambda)', default=1e-5)

args = parser.parse_args()

train_dataset = Mydata(r'D:\DARec\Dataset\ratings_Toys_and_Games.csv', r'D:\DARec\Dataset\ratings_Automotive.csv', train=True)
test_dataset = Mydata(r'D:\DARec\Dataset\ratings_Toys_and_Games.csv', r'D:\DARec\Dataset\ratings_Automotive.csv', train=False)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print("Data is loaded")

n_users, n_items = train_dataset.T_data.shape[0], train_dataset.T_data.shape[1]
model = AutoRec(n_users=n_users, n_items=n_items, n_factors=400).cuda()
criterion = MRMSELoss().cuda()

optimizer = optim.Adam(model.parameters(), weight_decay=args.wd, lr=args.lr)

def train(epoch):
    process = []
    for idx, d in enumerate(train_loader):
        data = d[1].cuda()
        optimizer.zero_grad()
        _, pred = model(data)
        pred.cuda()
        loss, mask = criterion(pred, data)
        RMSE = torch.sqrt(loss.item() / torch.sum(mask))
        loss.backward()
        optimizer.step()
        process.append(RMSE)
        if idx % 100 == 0:
            print (f"[+] Epoch {epoch} [{idx * args.batch_size} / {len(train_loader.dataset)}] - RMSE {sum(process) / len(process)}")
    return torch.tensor(sum(process) / (len(process))).cpu()


def test():
    process = []
    with torch.no_grad():
        for idx, d in enumerate(test_loader):
            data = d[1].cuda()
            optimizer.zero_grad()
            _, pred = model(data)
            pred.cuda()
            loss, mask = criterion(pred, data)
            RMSE = torch.sqrt(loss.item() / torch.sum(mask))
            # loss.backward()
            # optimizer.step()
            process.append(RMSE)
    print(f"[*] Test RMSE {sum(process) / len(process)} ")
    return torch.tensor(sum(process) / (len(process))).cpu()


if __name__=="__main__":
    train_rmse = []
    test_rmse = []
    wdir = r"D:\DARec\Pretrained_Parameters"

    for epoch in tqdm(range(args.epochs)):
        train_rmse.append(train(epoch))
        test_rmse.append(test())
        if epoch % 100 == 99:
            torch.save(model.state_dict(), wdir+"T_AutoRec_%d.pkl" % (epoch+1))
    plt.plot(range(args.epochs), train_rmse, range(args.epochs), test_rmse)
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.xticks(range(0, args.epochs, 2))
    plt.show()
