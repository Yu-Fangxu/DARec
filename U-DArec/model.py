import torch
import torch.nn as nn
from function import *
from AutoRec import U_AutoRec, I_AutoRec
import argparse
class U_DArec(nn.Module):
    def __init__(self, args):
        """
        args:
          n_users: int
          S_n_items: int
          T_n_items: int
          n_factors: int
          RPE_hidden_size: int
          is_source: bool     if input is sourse data
        """
        super(U_DArec, self).__init__()
        self.args = args
        self.n_factors = args.n_factors
        self.n_users = args.n_users
        self.S_n_items = args.S_n_items
        self.T_n_items = args.T_n_items
        self.S_autorec = U_AutoRec(self.n_users, self.S_n_items, self.n_factors)
        self.T_autorec = U_AutoRec(self.n_users, self.T_n_items, self.n_factors)
        # 加载预训练
        # self.S_autorec.load_state_dict(torch.load(args.S_pretrained_weights))
        # self.T_autorec.load_state_dict(torch.load(args.T_pretrained_weights))
        # 冻结预训练过的AutoRec参数
        for para in self.S_autorec.parameters():
            para.requires_grad = False
        for para in self.T_autorec.parameters():
            para.requires_grad = False
        self.RPE_hidden_size = args.RPE_hidden_size

        self.RPE = nn.Sequential(
            nn.Linear(self.n_factors, self.RPE_hidden_size),
            nn.ReLU()
            # nn.Linear(self.RPE_hidden_size, self.RPE_hidden_size),
            # nn.Sigmoid()
        )
        self.DC = nn.Sequential(
            nn.Linear(self.RPE_hidden_size, self.RPE_hidden_size // 2),
            nn.Sigmoid(),
            nn.Linear(self.RPE_hidden_size // 2, 2),
            nn.Sigmoid()
            # nn.Linear(self.RPE_hidden_size // 4, 2),
            # nn.Sigmoid()
        )
        self.S_RP = nn.Sequential(
            nn.Linear(self.RPE_hidden_size, self.RPE_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.RPE_hidden_size // 2, self.RPE_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.RPE_hidden_size // 2, self.S_n_items)
        )
        self.T_RP = nn.Sequential(
            nn.Linear(self.RPE_hidden_size, self.RPE_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.RPE_hidden_size // 2, self.RPE_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.RPE_hidden_size // 2, self.T_n_items)
        )

    def forward(self, rating_matrix, alpha, is_source):
        """
        rating_matrix: input matrix
        alpha: parameters in ReverseLayerF
        """
        rating_matrix.cuda()
        if is_source == True:
            embedding, _ = self.S_autorec(rating_matrix)
        else:
            embedding, _ = self.T_autorec(rating_matrix)
        feature = self.RPE(embedding)
        source_prediction = self.S_RP(feature)
        target_prediction = self.T_RP(feature)
        reversed_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.DC(reversed_feature)
        return class_output, source_prediction, target_prediction

if __name__ == "__main__":
    # 行是item，列是user
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_factors", type=int, default=800, help="embedding dim")
    parser.add_argument("--n_users", type=int, default=50, help="size of each image batch")
    parser.add_argument("--S_n_items", type=int, default=5000, help="learning rate")
    parser.add_argument("--T_n_items", type=int, default=5000, help="whether to apply data parallel")
    parser.add_argument("--RPE_hidden_size", type=int, default=500, help="whether to apply data parallel")
    args = parser.parse_args()
    x = torch.randn(1500, 5000)
    net = U_DArec(args)
    loss = DArec_Loss()
    class_output, source_prediction, target_prediction = net(x, 10, True)
    print(target_prediction.shape)
    print(class_output.shape)
    source_rating = source_prediction
    target_rating = target_prediction
    labels = torch.ones_like(class_output)[:, 1].long()
    labels2 = torch.ones(1500, 1).long()
    labels2 = labels2.squeeze(1)
    print(labels.shape)
    print(labels2.shape)
    print(loss(class_output, source_prediction, target_prediction, source_rating, target_rating, labels))
    for op in net.modules():
        print(op)
