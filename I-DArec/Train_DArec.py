import numpy as np
import torch.optim as optim
import torch.utils.data
from I_DArec import *
from torch.utils.data import DataLoader
from Data_Preprocessing import Mydata
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
parser = argparse.ArgumentParser(description='DArec with PyTorch')
parser.add_argument('--epochs', '-e', type=int, default=70)
parser.add_argument('--batch_size', '-b', type=int, default=64)
parser.add_argument('--lr', '-l', type=float, help='learning rate', default=1e-3)
parser.add_argument('--wd', '-w', type=float, help='weight decay(lambda)', default=1e-4)
parser.add_argument("--n_factors", type=int, default=200, help="embedding dim")
parser.add_argument("--n_users", type=int, default=1637, help="size of each image batch")
parser.add_argument("--S_n_items", type=int, default=23450, help="Source items number")
parser.add_argument("--T_n_items", type=int, default=16993, help="Target items number")
parser.add_argument("--RPE_hidden_size", type=int, default=200, help="hidden size of Rating Pattern Extractor")
parser.add_argument("--S_pretrained_weights", type=str, default=r'D:\DARec_I\Pretrained_ParametersS_AutoRec_50.pkl')
parser.add_argument("--T_pretrained_weights", type=str, default=r'D:\DARec_I\Pretrained_ParametersT_AutoRec_50.pkl')
args = parser.parse_args()

### Load Data
train_dataset = Mydata(r'D:\DARec\Dataset\ratings_Toys_and_Games.csv', r'D:\DARec\Dataset\ratings_Automotive.csv', train=True, preprocessed=True)
test_dataset = Mydata(r'D:\DARec\Dataset\ratings_Toys_and_Games.csv', r'D:\DARec\Dataset\ratings_Automotive.csv', train=False, preprocessed=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

args.n_users = train_dataset.S_data.shape[1]
args.S_n_items, args.T_n_items = train_dataset.S_data.shape[0], train_dataset.T_data.shape[0]

print("Data is loaded")
### neural network

net = I_DArec(args)
net.S_autorec.load_state_dict(torch.load(args.S_pretrained_weights))
net.T_autorec.load_state_dict(torch.load(args.T_pretrained_weights))
net.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), weight_decay=args.wd, lr=args.lr)
RMSE = MRMSELoss().cuda()
criterion = DArec_Loss().cuda()
# loss, source_mask, target_mask
def train(epoch):
    # process = []
    Total_RMSE = 0
    Total_MASK = 0
    for idx, d in enumerate(train_loader):
        ### 数据
        source_rating, target_rating, source_labels, target_labels = d

        source_rating = source_rating.cuda()
        target_rating = target_rating.cuda()
        source_labels = source_labels.squeeze(1).long().cuda()
        target_labels= target_labels.squeeze(1).long().cuda()

        optimizer.zero_grad()
        is_source = True
        if is_source == True:
            class_output, source_prediction, target_prediction = net(source_rating, is_source)

            source_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                    source_rating, target_rating, source_labels)

            rmse, _ = RMSE(source_prediction, source_rating)
            # rmse, _ = RMSE(source_prediction, source_rating)

            Total_RMSE += rmse.item()
            Total_MASK += torch.sum(target_mask).item()

            loss = source_loss
        is_source = False
        if is_source == False:
            class_output, source_prediction, target_prediction = net(target_rating, is_source)
            target_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                              source_rating, target_rating, target_labels)
            loss += target_loss

        loss.backward()
        optimizer.step()

    #     process.append(rmse)
    # return torch.tensor(sum(process) / (len(process))).clone().detach().cpu()
    return math.sqrt(Total_RMSE / Total_MASK)

def test():
    Total_RMSE = 0
    Total_MASK = 0
    with torch.no_grad():
        for idx, d in enumerate(test_loader):
            ### 数据
            source_rating, target_rating, source_labels, target_labels = d

            source_rating = source_rating.cuda()
            target_rating = target_rating.cuda()
            source_labels = source_labels.squeeze(1).long().cuda()
            target_labels = target_labels.squeeze(1).long().cuda()

            # optimizer.zero_grad()
            is_source = True
            if is_source == True:
                class_output, source_prediction, target_prediction = net(source_rating, is_source)
                source_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
                                                        source_rating, target_rating, source_labels)
                rmse, _ = RMSE(source_prediction, source_rating)
                #rmse, _ = RMSE(source_prediction, source_rating)
                Total_RMSE += rmse.item()
                Total_MASK += torch.sum(target_mask).item()

                loss = source_loss
            # is_source = False
            # if is_source == False:
            #     class_output, source_prediction, target_prediction = net(target_rating, alpha, is_source)
            #     target_loss, source_mask, target_mask = criterion(class_output, source_prediction, target_prediction,
            #                                                       source_rating, target_rating, source_class)
            #     loss += target_loss


    return math.sqrt(Total_RMSE / Total_MASK)

if __name__=="__main__":
    train_rmse = []
    test_rmse = []
    wdir = r"D:\DARec\DARec_"

    for epoch in tqdm(range(args.epochs)):
        train_rmse.append(train(epoch))
        test_rmse.append(test())
        if epoch % args.epochs == args.epochs - 1:
            torch.save(net.state_dict(), wdir+"%d.pkl" % (epoch+1))
    print(min(test_rmse))
    plt.plot(range(args.epochs), train_rmse, range(args.epochs), test_rmse)
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.xticks(range(0, args.epochs, 2))
    plt.show()
