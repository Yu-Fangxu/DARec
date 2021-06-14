from torch.utils.data import DataLoader, Dataset
from collections import Counter
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
class Mydata(Dataset):
    def __init__(self, S_path, T_path, train_ratio=0.9, test_ratio=0.1, train=None, ):
        super().__init__()
        self.S_path = S_path
        self.T_path = T_path
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        S_df = pd.read_csv(S_path,header=None)
        T_df = pd.read_csv(T_path, header=None)
        S_df.columns = ["User", "Item", "Rating", "TimeStamp"]
        T_df.columns = ["User", "Item", "Rating", "TimeStamp"]

        S_cnt = Counter(S_df.iloc[:, 0])
        S_cnt = {k: v for k, v in S_cnt.items() if v >= 5}
        T_cnt = Counter(T_df.iloc[:, 0])
        T_cnt = {k: v for k, v in T_cnt.items() if v >= 5}
        ### 求两个人名的交集
        S_user = set(S_cnt.keys())
        T_user = set(T_cnt.keys())
        users = list(S_user.intersection(T_user))
        S_df = S_df.loc[S_df["User"].isin(users)]
        T_df = T_df.loc[T_df["User"].isin(users)]
        ### 所有用户对应一个序号
        dict_users = {users[i]: i for i in range(len(users))}
        ### 交集4075人
        S_items = list(set(S_df.iloc[:, 1]))

        T_items = list(set(T_df.iloc[:, 1]))
        ### 所有数据对应一个序号
        # print(len(T_items))
        dict_S_items = {S_items[i]: i for i in range(len(S_items))}
        dict_T_items = {T_items[i]: i for i in range(len(T_items))}
        S_df.reset_index(drop=True, inplace=True)
        T_df.reset_index(drop=True, inplace=True)
        ### 将代码转换为序号
        for index, row in tqdm(S_df.iterrows()):
            user_idx = dict_users[row["User"]]
            item_idx = dict_S_items[row["Item"]]
            S_df.iloc[index, 0] = user_idx
            S_df.iloc[index, 1] = item_idx
        for index, row in tqdm(T_df.iterrows()):
            user_idx = dict_users[row["User"]]
            item_idx = dict_T_items[row["Item"]]
            T_df.iloc[index, 0] = user_idx
            T_df.iloc[index, 1] = item_idx
        # 样例
        # Item    User    Rating
        # 22      32          5
        # print(S_df.head())
        print(len(users))

        self.S_data = torch.zeros((len(users), len(S_items)))
        self.T_data = torch.zeros((len(users), len(T_items)))
        for index, row in tqdm(S_df.iterrows()):
            user = row["User"]
            item = row["Item"]
            self.S_data[user, item] = row["Rating"]

        for index, row in tqdm(T_df.iterrows()):
            user = row["User"]
            item = row["Item"]
            self.T_data[user, item] = row["Rating"]

        self.S_y = torch.zeros((self.S_data.shape[1], 1))
        self.T_y = torch.ones((self.T_data.shape[1], 1))
        self.total_indices = np.arange(len(self.S_data))
        self.test_indices = np.random.choice(self.total_indices, size=
                            int(len(self.S_data) * self.test_ratio), replace=False)
        self.train_indices = np.array(list(set(self.total_indices)-set(self.test_indices)))

        if train != None:
            if train == True:
                self.S_data = self.S_data[self.train_indices]
                self.T_data = self.T_data[self.train_indices]
                self.S_y = self.S_y[self.train_indices]
                self.T_y = self.T_y[self.train_indices]
            else:
                self.S_data = self.S_data[self.test_indices]
                self.T_data = self.T_data[self.test_indices]
                self.S_y = self.S_y[self.test_indices]
                self.T_y = self.T_y[self.test_indices]
        # print(self.S_data.shape)
        # print(self.T_data.shape)
        # print(self.S_y.shape)
        # print(self.T_y.shape)
    def __getitem__(self, item):
        return (self.S_data[item], self.T_data[item], self.S_y[item], self.T_y[item])

    def __len__(self):
        return self.S_data.shape[0]

if __name__ == "__main__":
    data = Mydata(r'D:\DARec\Dataset\ratings_Toys_and_Games.csv',
                  r'D:\DARec\Dataset\ratings_Automotive.csv', train=True)

    dataloader = DataLoader(data, batch_size=8, shuffle=True)
    for d in dataloader:
        print(d[3])
        break