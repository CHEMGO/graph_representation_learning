import numpy as np
import torch
import torch.nn.functional as F
import pickle
import pandas as pd
from sklearn import preprocessing

from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from utils.plot_utils import plot_curve, plot_sample
from utils.utils import build_optimizer, objectview, mask_edge
from preprocess import get_known_mask,get_train_mask

def train_gnn_mdi(data, args, log_path, device=torch.device('cpu')):
    model = get_gnn(data, args).to(device)
    if args.impute_hiddens == '':
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int,args.impute_hiddens.split('_')))
    if args.concat_states:
        input_dim = args.node_dim * len(model.convs) * 2
    else:
        input_dim = args.node_dim * 2
    if hasattr(args,'ce_loss') and args.ce_loss:
        output_dim = len(data.class_values)
    else:
        output_dim = 1
    impute_model = MLPNet(input_dim, output_dim,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            dropout=args.dropout).to(device)
    if args.transfer_dir: # this ensures the valid mask is consistant
        load_path = './{}/test/{}/{}/'.format(args.domain,args.data,args.transfer_dir)
        print("loading fron {} with {}".format(load_path,args.transfer_extra))
        model = torch.load(load_path+'model'+args.transfer_extra+'.pt',map_location=device)
        impute_model = torch.load(load_path+'impute_model'+args.transfer_extra+'.pt',map_location=device)

    trainable_parameters = list(model.parameters()) \
                           + list(impute_model.parameters())
    print("total trainable_parameters: ",len(trainable_parameters))
    # build optimizer
    scheduler, opt = build_optimizer(args, trainable_parameters)

    # train
    Train_loss = []
    Test_rmse = []
    Test_l1 = []
    Lr = []


    x = data.x.clone().detach().to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    # print('edge_attr length is:',len(edge_attr))
    edge_labels = data.edge_labels.to(device)
    all_train_edge_index = data.train_edge_index.clone().detach().to(device)
    all_train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    all_train_labels = data.train_labels.clone().detach().to(device)
    if hasattr(data,'class_values'):
        class_values = data.class_values.clone().detach().to(device)
    if args.valid > 0.:
        valid_mask = get_train_mask(args.valid, int(all_train_edge_attr.shape[0] / 2)).to(device)
        print("valid mask sum: ",torch.sum(valid_mask))
        print(len(valid_mask))
        train_labels = all_train_labels[~valid_mask]
        valid_labels = all_train_labels[valid_mask]
        double_valid_mask = torch.cat((valid_mask, valid_mask), dim=0)
        valid_edge_index, valid_edge_attr = mask_edge(all_train_edge_index, all_train_edge_attr, double_valid_mask, True)
        train_edge_index, train_edge_attr = mask_edge(all_train_edge_index, all_train_edge_attr, ~double_valid_mask, True)
        print("train edge num is {}, valid edge num is {}"\
                .format(
                train_edge_attr.shape[0], valid_edge_attr.shape[0]))
        Valid_rmse = []
        Valid_l1 = []
        best_valid_rmse = np.inf
        best_valid_rmse_epoch = 0
        best_valid_l1 = np.inf
        best_valid_l1_epoch = 0
    else:
        train_edge_index, train_edge_attr, train_labels =\
             all_train_edge_index, all_train_edge_attr, all_train_labels
        print("train edge num is {}"\
                .format(
                train_edge_attr.shape[0]))

    print("train edge num is {}"\
            .format(
            train_edge_attr.shape[0]))
    obj = dict()
    obj['args'] = args
    obj['outputs'] = dict()
    for epoch in range(args.epochs):
        print('epoch: ', epoch)
        model.train()
        impute_model.train()

        print('length of train_attr is:')
        print(len(train_edge_attr))
        known_mask = get_train_mask(args.known, int(train_edge_attr.shape[0] / 2)).to(device)
        print(len(known_mask))
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)
        # known_edge_index = data.train_edge_index
        # known_edge_attr = data.train_edge_attr
        print('length of known_edge_attr is',len(known_edge_attr))

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)


        print(len(x_embd[0]))
        print('shape of x_embd is:')
        print(x_embd.shape)
        print('length of x_embd is:')
        print(len(x_embd))
        # print(x_embd[1])
        print('train_edge index:')
        print(train_edge_index[0])
        print(len(train_edge_index[0]))
        print(train_edge_index[1])



        pred = impute_model([x_embd[train_edge_index[0]], x_embd[train_edge_index[1]]])





        # print(pred)
        print('pred length is: ')
        print(len(pred))
        print(pred)

        pred_train = pred[:int(train_edge_attr.shape[0] / 2)]
        print(pred_train)
        # pred_train = np.array(pred_train).reshape(-1, 8)
        # min_max_scaler = data.min_max_scaler
        # # origin_labels = origin_labels.reshape(-1,1)
        # pred_train = min_max_scaler.inverse_transform(pred_train)
        # print('predict train is :')
        # print(pred_train)




        if hasattr(args,'ce_loss') and args.ce_loss:
            pred_train = pred[:int(train_edge_attr.shape[0] / 2)]
        else:
            pred_train = pred[:int(train_edge_attr.shape[0] / 2),0]
        if args.loss_mode == 1:
            pred_train[known_mask] = train_labels[known_mask]
        label_train = train_labels

        if hasattr(args,'ce_loss') and args.ce_loss:
            loss = F.cross_entropy(pred_train,train_labels)
        else:
            loss = F.mse_loss(pred_train, label_train)
        loss.backward()
        opt.step()
        train_loss = loss.item()
        if scheduler is not None:
            scheduler.step(epoch)
        for param_group in opt.param_groups:
            Lr.append(param_group['lr'])



        model.eval()
        impute_model.eval()


        with torch.no_grad():
            if args.valid > 0.:
                x_embd = model(x, train_edge_attr, train_edge_index)
                pred = impute_model([x_embd[valid_edge_index[0], :], x_embd[valid_edge_index[1], :]])
                if hasattr(args,'ce_loss') and args.ce_loss:
                    pred_valid = class_values[pred[:int(valid_edge_attr.shape[0] / 2)].max(1)[1]]
                    label_valid = class_values[valid_labels]
                elif hasattr(args,'norm_label') and args.norm_label:
                    pred_valid = pred[:int(valid_edge_attr.shape[0] / 2),0]
                    pred_valid = pred_valid * max(class_values)
                    label_valid = valid_labels
                    label_valid = label_valid * max(class_values)
                else:
                    pred_valid = pred[:int(valid_edge_attr.shape[0] / 2),0]
                    label_valid = valid_labels
                mse = F.mse_loss(pred_valid, label_valid)
                valid_rmse = np.sqrt(mse.item())
                l1 = F.l1_loss(pred_valid, label_valid)
                valid_l1 = l1.item()
                if valid_l1 < best_valid_l1:
                    best_valid_l1 = valid_l1
                    best_valid_l1_epoch = epoch
                    if args.save_model:
                        torch.save(model.state_dict(), log_path + 'model_best_valid_l1.pt')
                        torch.save(impute_model.state_dict(), log_path + 'impute_model_best_valid_l1.pt')
                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_valid_rmse_epoch = epoch
                    if args.save_model:
                        torch.save(model, log_path + 'model_best_valid_rmse.pt')
                        torch.save(impute_model, log_path + 'impute_model_best_valid_rmse.pt')
                Valid_rmse.append(valid_rmse)
                Valid_l1.append(valid_l1)


    #
    # pred_train = pred_train.detach().cpu().numpy()
    # label_train = label_train.detach().cpu().numpy()
    # pred_test = pred_test.detach().cpu().numpy()
    # label_test = label_test.detach().cpu().numpy()
    #
    # obj['curves'] = dict()
    # obj['curves']['train_loss'] = Train_loss
    # if args.valid > 0.:
    #     obj['curves']['valid_rmse'] = Valid_rmse
    #     obj['curves']['valid_l1'] = Valid_l1
    # obj['curves']['test_rmse'] = Test_rmse
    # obj['curves']['test_l1'] = Test_l1
    # obj['lr'] = Lr
    #
    # obj['outputs']['final_pred_train'] = pred_train
    # obj['outputs']['label_train'] = label_train
    # obj['outputs']['final_pred_test'] = pred_test
    # obj['outputs']['label_test'] = label_test
    # pickle.dump(obj, open(log_path + 'result.pkl', "wb"))
    #
    # if args.save_model:
    #     torch.save(model, log_path + 'model.pt')
    #     torch.save(impute_model, log_path + 'impute_model.pt')
    #
    # # obj = objectview(obj)
    # plot_curve(obj['curves'], log_path+'curves.png',keys=None,
    #             clip=True, label_min=True, label_end=True)
    # plot_curve(obj, log_path+'lr.png',keys=['lr'],
    #             clip=False, label_min=False, label_end=False)
    # plot_sample(obj['outputs'], log_path+'outputs.png',
    #             groups=[['final_pred_train','label_train'],
    #                     ['final_pred_test','label_test']
    #                     ],
    #             num_points=20)
    # if args.save_prediction and args.valid > 0.:
    #     plot_sample(obj['outputs'], log_path+'outputs_best_valid.png',
    #                 groups=[['best_valid_rmse_pred_test','label_test'],
    #                         ['best_valid_l1_pred_test','label_test']
    #                         ],
    #                 num_points=20)
    # if args.valid > 0.:
    #     print("best valid rmse is {:.3g} at epoch {}".format(best_valid_rmse,best_valid_rmse_epoch))
    #     print("best valid l1 is {:.3g} at epoch {}".format(best_valid_l1,best_valid_l1_epoch))
    #
    #
    # print('predict train is:')
    # print(pred_train)
    # print(len(pred_train))
    # print('label train is:')
    # print(label_train)
    # print('predict test is:')
    # print(pred_test)
    # print('label test is:')
    # print(label_test)

    # torch.save(model,'model_best_valid_rmse.pt')
    # torch.save(impute_model, 'impute_model_best_valid_rmse.pt')
    model.eval()
    impute_model.eval()
    print('train is over! now its final result:')
    known_index = data.train_edge_index.clone().detach().to(device)
    known_attr = data.train_edge_attr.clone().detach().to(device)
    x_embd = model(x, known_attr, known_index)
    pred = impute_model([x_embd[edge_index[0]], x_embd[edge_index[1]]])

    pred = pred[:int(edge_attr.shape[0] / 2)]
    print(len(pred))
    pred = pred.cpu().detach().numpy().reshape(-1, 9)
    min_max_scaler = data.min_max_scaler
    # origin_labels = origin_labels.reshape(-1,1)
    pred_origin = min_max_scaler.inverse_transform(pred)
    print('predict train is :')
    print(pred_origin[0])



    df_y = data.df_y
    df_class = data.df_class
    print(len(pred_origin))
    print(len(df_y))
    # pred_origin = np.append(pred_origin, df_y, axis=1)  # axis=1表示对应行的数组进行拼接
    pred_origin = np.column_stack((pred_origin,df_class,df_y))

    # 将结果写入csv文件里
    pd_data = pd.DataFrame(pred_origin, columns=["gender","age","bmi","bloodGlucose","proinsulin","Cp120","diabetesPredigreeFunction","trainDataSource","trainOutcome","dataSource","outcome"])
    pd_data.to_csv('pd_data_y.csv',index = False, float_format = '%.04f')

    #将结果替代原数据中缺失的部分
    df_X = data.df_X
    df_X = min_max_scaler.inverse_transform(df_X)
    print(df_X[0])
    nrow, ncol = df_X.shape
    for i in range(nrow):
        for j in range(ncol):
            if np.isnan(df_X[i][j]):
                df_X[i][j] = pred_origin[i][j]
    # df = np.concatenate((df_X, df_y), axis=1)  # axis=1表示对应行的数组进行拼接
    df = np.column_stack((df_X,df_class,df_y))
    pd_data_origin = pd.DataFrame(df,columns=["gender", "age", "bmi", "bloodGlucose", "proinsulin", "Cp120", "diabetesPredigreeFun","trainDataSource","trainOutcome","dataSource","outcome"])
    pd_data_origin.to_csv("pd_data_origin_y.csv",index = False, float_format = '%.04f')