import torch
import torch.nn as nn
from function import *
import matplotlib.pyplot as plt
class U_AutoRec(nn.Module):
    """
    input -> hidden -> output(output.shape == input.shape)
    encoder: input -> hidden
    decoder: hidden -> output
    """

    def __init__(self, n_users, n_items, n_factors=800, p_drop=0.1):
        super(U_AutoRec, self).__init__()
        self.n_factors = n_factors
        self.n_users = n_users
        self.n_items = n_items
        self.encoder = nn.Sequential(
            nn.Linear(self.n_items, self.n_factors),
            # nn.Dropout(p_drop),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_factors, self.n_items),
            # nn.Dropout(p_drop)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding, self.decoder(embedding)

class I_AutoRec(nn.Module):
    """
    input -> hidden -> output(output.shape == input.shape)
    encoder: input -> hidden
    decoder: hidden -> output
    """

    def __init__(self, n_users, n_items, n_factors=800, p_drop=0.25):
        super(I_AutoRec, self).__init__()
        self.n_factors = n_factors
        self.n_users = n_users
        self.n_items = n_items
        self.encoder = nn.Sequential(
            nn.Linear(self.n_users, self.n_factors),
            nn.Dropout(p_drop),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_factors, self.n_users),
            nn.Dropout(p_drop),
            #nn.Identity(),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding, self.decoder(embedding)


if __name__ == "__main__":
    # 行是item，列是user
    # x = torch.randn(4, 1500, 5000)
    # net = U_AutoRec(1500, 5000)
    # loss = MRMSELoss()
    # embedding, y = net(x)
    # print(y.shape)
    # print(embedding.shape)
    # print(loss(x, y).data)


    # names = ['200', '400', '600', '800', '1000', '1200']
    # x = range(len(names))
    # y1 = [3.648, 3.644, 3.637, 3.635, 3.638, 3.636] #Automotive
    # y2 = [3.56, 3.59, 3.61, 3.61, 3.57, 3.59] #CDs and Vinyl
    # y3 = [2.6, 2.58, 2.581, 2.571, 2.56, 2.55] #Video nad games
    # y4 = [2.2, 2.23, 2.22, 2.24, 2.23, 2.22] #movies
    # plt.plot(x, y1, marker='o', mec='r', label=u'Automotive')
    # plt.plot(x, y2, marker='*', ms=10, label=u'CDs and Vinyl')
    # plt.plot(x, y3, marker='x', ms=10, label=u'Video and Games')
    # plt.plot(x, y4, marker='.', ms=10, label=u'movies')
    # plt.legend()  # 让图例生效
    # plt.xticks(x, names, rotation=45)
    # plt.margins(0)
    # plt.subplots_adjust(bottom=0.15)
    # plt.xlabel(u"Embedding size")  # X轴标签
    # plt.ylabel("RMSE")  # Y轴标签
    # plt.title("A simple plot")  # 标题
    # plt.show()

    # name_list = ['U_AutoRec', 'DArec']
    # num_list = [3.58, 3.8]
    # plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list)
    # plt.xlabel("Models")
    # plt.ylabel("RMSE")
    # # plt.title("Office Products & Movies and TV")
    # # plt.title("Sports and Outdoors & CDs and Vinyl")
    # # plt.title("Android and Apps & Video Games")
    # plt.title("Toys and Games & Automotive")
    # plt.show()

    # name_list = ["Toys & Automotive", "Sports & CD", "Apps & Video Games", "Office & Movies"]
    # num_list = [0.71, 3.56, 2.54, 2.17]
    # plt.bar(range(len(num_list)), num_list, tick_label=name_list)
    # plt.xlabel("Datasets")
    # plt.ylabel("RMSE")
    # plt.title("Cross Domain Recommendation")
    # plt.show()
    #
    name_list = ["Toys & Automotive", "Sports & CD", "Apps & Video Games", "Office & Movies"]
    num_list = [0.83, 0.342, 1.2, 0.45]
    plt.bar(range(len(num_list)), num_list, tick_label=name_list, width=0.5)
    plt.xlabel("Datasets")
    plt.ylabel("RMSE")
    plt.title("Single Domain")
    plt.show()
