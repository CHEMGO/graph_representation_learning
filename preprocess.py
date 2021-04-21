import pandas as pd
import os.path as osp
import inspect
from torch_geometric.data import Data
from sklearn import preprocessing
from pandas import read_csv
import torch
import random
import numpy as np
import pdb


def create_node(df, mode):
    if mode == 0:  # onehot feature node, all 1 sample node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol, ncol))
        feature_node[np.arange(ncol), feature_ind] = 1  # 生成ncol * ncol的单位矩阵
        sample_node = [[1] * ncol for i in range(nrow)]
        node = sample_node + feature_node.tolist()
    elif mode == 1:  # onehot sample and feature node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol, ncol + 1))
        feature_node[np.arange(ncol), feature_ind + 1] = 1
        sample_node = np.zeros((nrow, ncol + 1))
        sample_node[:, 0] = 1
        node = sample_node.tolist() + feature_node.tolist()
    return node


def create_edge(df):
    n_row, n_col = df.shape
    edge_start = []
    edge_end = []
    for x in range(n_row):
        edge_start = edge_start + [x] * n_col  # obj
        edge_end = edge_end + list(n_row + np.arange(n_col))  # att
    edge_start_new = edge_start + edge_end
    edge_end_new = edge_end + edge_start
    return (edge_start_new, edge_end_new)


def create_edge_attr(df):
    nrow, ncol = df.shape
    edge_attr = []
    for i in range(nrow):
        for j in range(ncol):
            edge_attr.append([float(df.iloc[i, j])])
    edge_attr = edge_attr + edge_attr
    return edge_attr


def get_known_mask(df):
    train_edge_mask = []
    nrow,ncol = df.shape
    for i in range(nrow):
        for j in range(ncol):
            n = df.iloc[i,j]
            if  pd.isna(df.iloc[i, j]):
                train_edge_mask.append(False)
            else:
                train_edge_mask.append(True)
    return torch.BoolTensor(train_edge_mask)


def get_train_mask(known_prob, edge_num):
    known_mask = (torch.FloatTensor(edge_num, 1).uniform_() < known_prob).view(-1)
    return known_mask


def mask_edge(edge_index,edge_attr,mask,remove_edge):
    edge_index = edge_index.clone().detach()
    edge_attr = edge_attr.clone().detach()
    if remove_edge:
        edge_index = edge_index[:,mask]
        edge_attr = edge_attr[mask]
    else:
        edge_attr[~mask] = 0.
    return edge_index, edge_attr



def get_data(df_X, df_y,df_class,node_mode, seed=0,
             normalize=True):
    if len(df_y.shape) == 1:
        df_y = df_y.to_numpy()
    elif len(df_y.shape) == 2:
        df_y = df_y[0].to_numpy()

    # keep train_edge_prob of all edges
    train_edge_mask = get_known_mask(df_X)
    double_train_edge_mask = torch.cat((train_edge_mask, train_edge_mask), dim=0)

    if normalize:
        x = df_X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_X = pd.DataFrame(x_scaled)
        print('x_scaled is :')
        print(df_X.iloc[0])
        # df_X = pd.DataFrame(x)

    edge_start, edge_end = create_edge(df_X)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    print('length of edge_index is: %d',len(edge_index[0]))
    edge_attr = torch.tensor(create_edge_attr(df_X), dtype=torch.float)
    print('edge_attr is:')
    print(edge_attr)
    node_init = create_node(df_X, node_mode)
    x = torch.tensor(node_init, dtype=torch.float)
    y = torch.tensor(df_y, dtype=torch.float)

    # set seed to fix known/unknwon edges
    torch.manual_seed(seed)


    # mask edges based on the generated train_edge_mask
    # train_edge_index is known, test_edge_index in unknwon, i.e. missing
    train_edge_index, train_edge_attr = mask_edge(edge_index, edge_attr,
                                                  double_train_edge_mask, True)

    print('length of double_mask is: %d', len(double_train_edge_mask))
    print('length of train_edge_index is: %d', len(train_edge_index[0]))
    train_labels = train_edge_attr[:int(train_edge_attr.shape[0] / 2), 0]
    edge_labels = edge_attr[:int(edge_attr.shape[0] / 2), 0]

#    train_y_mask = get_known_mask(train_y_prob, y.shape[0])
#    test_y_mask = ~train_y_mask

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
                train_edge_index=train_edge_index, edge_labels=edge_labels,
                train_edge_attr=train_edge_attr,
                train_edge_mask=train_edge_mask, train_labels=train_labels,
                df_X=df_X, df_y=df_y, df_class =df_class,
                edge_attr_dim=train_edge_attr.shape[-1],
                user_num=df_X.shape[0],
                min_max_scaler=min_max_scaler,

                )
    return data


def load_data(args):
    dataset = read_csv('test1.csv')
    array = dataset.values
    df_y = pd.DataFrame(array[:, -1:])
    df_class = pd.DataFrame(array[:,8])
    df_X = pd.DataFrame(np.column_stack((array[:, 1:8],array[:,9])))
    print('feature example:')
    print(df_X)
    data = get_data(df_X, df_y, df_class,args.node_mode,args.seed)
    return data


if __name__ == '__main__':
    data = load_data()

