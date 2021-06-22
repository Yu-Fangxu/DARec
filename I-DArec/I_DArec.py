import torch
import torch.nn as nn
from function import *
import argparse
from AutoRec import I_AutoRec
class I_DArec(nn.Module):
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

        super(I_DArec, self).__init__()
        self.args = args
        self.n_factors = args.n_factors
        self.n_users = args.n_users
        self.S_n_items = args.S_n_items
        self.T_n_items = args.T_n_items
        self.S_autorec = I_AutoRec(self.n_users, self.S_n_items, self.n_factors)
        self.T_autorec = I_AutoRec(self.n_users, self.T_n_items, self.n_factors)
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
            nn.Linear(self.RPE_hidden_size // 2, self.n_users)
        )
        self.T_RP = nn.Sequential(
            nn.Linear(self.RPE_hidden_size, self.RPE_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.RPE_hidden_size // 2, self.RPE_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.RPE_hidden_size // 2, self.n_users)
        )

    def forward(self, rating_matrix, is_source):
        """
        rating_matrix: input matrix
        alpha: parameters in ReverseLayerF
        """
        if is_source == True:
            embedding, _ = self.S_autorec(rating_matrix)
        else:
            embedding, _ = self.T_autorec(rating_matrix)
        feature = self.RPE(embedding)
        source_prediction = self.S_RP(feature)
        target_prediction = self.T_RP(feature)
        class_output = self.DC(feature)
        return class_output, source_prediction, target_prediction