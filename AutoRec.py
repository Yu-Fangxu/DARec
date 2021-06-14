import torch
import torch.nn as nn
from function import *
class AutoRec(nn.Module):
    """
    input -> hidden -> output(output.shape == input.shape)
    encoder: input -> hidden
    decoder: hidden -> output
    """

    def __init__(self, n_users, n_items, n_factors=800):
        super(AutoRec, self).__init__()
        self.n_factors = n_factors
        self.n_users = n_users
        self.n_items = n_items
        self.encoder = nn.Sequential(
            nn.Linear(self.n_items, self.n_factors),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_factors, self.n_items),
            nn.Identity(),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding, self.decoder(embedding)

if __name__ == "__main__":
    # 行是item，列是user
    x = torch.randn(4, 1500, 5000)
    net = AutoRec(1500, 5000)
    loss = MRMSELoss()
    embedding, y = net(x)
    print(y.shape)
    print(embedding.shape)
    print(loss(x, y).data)
